import torch
from botorch.optim.optimize import optimize_acqf
from botorch.utils.multi_objective.box_decompositions.non_dominated import FastNondominatedPartitioning
from botorch.acquisition.multi_objective.monte_carlo import qExpectedHypervolumeImprovement
from botorch.utils.transforms import normalize
from botorch.utils.safe_math import log_fatplus
from botorch.utils.multi_objective.hypervolume import Hypervolume
from botorch.acquisition.multi_objective.monte_carlo import MultiObjectiveMCAcquisitionFunction
from botorch.utils.multi_objective.box_decompositions.non_dominated import NondominatedPartitioning
from botorch.acquisition.multi_objective.objective import MCMultiOutputObjective
from botorch.sampling.base import MCSampler
from botorch.utils.transforms import concatenate_pending_points, t_batch_mode_transform
from botorch.models.model import Model
from botorch.utils.objective import compute_smoothed_feasibility_indicator
from botorch.utils.multi_objective.hypervolume import SubsetIndexCachingMixin
from botorch.exceptions import BotorchTensorDimensionError, UnsupportedError
from botorch.optim.initializers import sample_truncated_normal_perturbations
from torch import Tensor
from typing import List, Optional, Union, Callable

from mobo_fixed_size.variable import NUM_RESTARTS,RAW_SAMPLES,OPTIONS,BATCH_SIZE



###################### SMS-EGO #############################
class UCB_HV(MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
        fat: bool = False,
    ) -> None:
      
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        SubsetIndexCachingMixin.__init__(self)
        self.fat = fat

    def _compute_qehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
      
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = compute_smoothed_feasibility_indicator(
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
                fat=self.fat,
            )  # `sample_shape x batch-shape x q`
        device = self.ref_point.device
        q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # subtract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1).mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:

        beta1 = 2.
        # beta2 = -1.

        posterior = self.model.posterior(X)

        dim = posterior.mean.shape[-1]

      
        samples1 = posterior.mean + beta1*posterior.variance.sqrt()
        samples1 = samples1.reshape(1,-1,1,dim)



        samples = samples1

        return self._compute_qehvi(samples=samples, X=X)
    

###################### CHO-EI #############################
class qExpectedHypervolumeImprovement_FixedSizedParetoFront_full(
    MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin
):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        pareto_front: Tensor,  # this is the pareto front of size M (current best)
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
        fat: bool = False,
        mean: bool = False,
        log: bool = False

    ) -> None:

        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )


        #use this to check EI method
        hv_computer_temp = Hypervolume(ref_point)
        self.best_HV = hv_computer_temp.compute(pareto_front)


        self.pareto_front = pareto_front


        self.mean = mean

        self.log = log


        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        SubsetIndexCachingMixin.__init__(self)
        self.fat = fat

    def _compute_qehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:

        # Note that the objective may subset the outcomes (e.g. this will usually happen if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = compute_smoothed_feasibility_indicator(
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
                fat=self.fat,
            )  # `sample_shape x batch-shape x q`
        device = self.ref_point.device
        q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # subtract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples

        if self.log:
            Z = areas_per_segment.sum(dim=-1) - self.best_HV
            temp = log_fatplus(Z, tau=1e-6) 

        
        else:
            if not self.mean:
                temp = (areas_per_segment.sum(dim=-1) - self.best_HV).clamp_min(0.0)

            elif self.mean:
                temp = areas_per_segment.sum(dim=-1) 


        return temp.mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)
        return self._compute_qehvi(samples=samples, X=X)


###################### CHO-UCB #############################
class UCB_HypervolumeImprovement(
    MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin
):
    def __init__(
        self,
        beta,
        model: Model,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
        fat: bool = False,
    ) -> None:
      
        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.register_buffer("ref_point", ref_point)
        cell_bounds = partitioning.get_hypercell_bounds()
        self.register_buffer("cell_lower_bounds", cell_bounds[0])
        self.register_buffer("cell_upper_bounds", cell_bounds[1])
        SubsetIndexCachingMixin.__init__(self)
        self.fat = fat

        self.beta = beta

    def _compute_qehvi(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:
        r"""Compute the expected (feasible) hypervolume improvement given MC samples.

        Args:
            samples: A `n_samples x batch_shape x q' x m`-dim tensor of samples.
            X: A `batch_shape x q x d`-dim tensor of inputs.

        Returns:
            A `batch_shape x (model_batch_shape)`-dim tensor of expected hypervolume
            improvement for each batch.
        """
        # Note that the objective may subset the outcomes (e.g. this will usually happen
        # if there are constraints present).
        obj = self.objective(samples, X=X)
        q = obj.shape[-2]
        if self.constraints is not None:
            feas_weights = compute_smoothed_feasibility_indicator(
                constraints=self.constraints,
                samples=samples,
                eta=self.eta,
                fat=self.fat,
            )  # `sample_shape x batch-shape x q`
        device = self.ref_point.device
        q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
        batch_shape = obj.shape[:-2]
        # this is n_samples x input_batch_shape x
        areas_per_segment = torch.zeros(
            *batch_shape,
            self.cell_lower_bounds.shape[-2],
            dtype=obj.dtype,
            device=device,
        )
        cell_batch_ndim = self.cell_lower_bounds.ndim - 2
        sample_batch_view_shape = torch.Size(
            [
                batch_shape[0] if cell_batch_ndim > 0 else 1,
                *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                *self.cell_lower_bounds.shape[1:-2],
            ]
        )
        view_shape = (
            *sample_batch_view_shape,
            self.cell_upper_bounds.shape[-2],
            1,
            self.cell_upper_bounds.shape[-1],
        )
        for i in range(1, self.q_out + 1):
            # TODO: we could use batches to compute (q choose i) and (q choose q-i)
            # simultaneously since subsets of size i and q-i have the same number of
            # elements. This would decrease the number of iterations, but increase
            # memory usage.
            q_choose_i = q_subset_indices[f"q_choose_{i}"]
            # this tensor is mc_samples x batch_shape x i x q_choose_i x m
            obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
            obj_subsets = obj_subsets.view(
                obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
            )
            # since all hyperrectangles share one vertex, the opposite vertex of the
            # overlap is given by the component-wise minimum.
            # take the minimum in each subset
            overlap_vertices = obj_subsets.min(dim=-2).values
            # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
            # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
            overlap_vertices = torch.min(
                overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
            )
            # subtract cell lower bounds, clamp min at zero
            lengths_i = (
                overlap_vertices - self.cell_lower_bounds.view(view_shape)
            ).clamp_min(0.0)
            # take product over hyperrectangle side lengths to compute area
            # sum over all subsets of size i
            areas_i = lengths_i.prod(dim=-1)
            # if constraints are present, apply a differentiable approximation of
            # the indicator function
            if self.constraints is not None:
                feas_subsets = feas_weights.index_select(
                    dim=-1, index=q_choose_i.view(-1)
                ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
            areas_i = areas_i.sum(dim=-1)
            # Using the inclusion-exclusion principle, set the sign to be positive
            # for subsets of odd sizes and negative for subsets of even size
            areas_per_segment += (-1) ** (i + 1) * areas_i
        # sum over segments and average over MC samples
        return areas_per_segment.sum(dim=-1).mean(dim=0)

    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        # samples = self.get_posterior_samples(posterior)

        mean_temp = posterior.mean
        std_temp = posterior.variance.sqrt()

        ucb_value = mean_temp + self.beta*std_temp

        ucb_value = ucb_value.unsqueeze(0)

        # print('ucb_shape ',ucb_value.shape)


        return self._compute_qehvi(samples=ucb_value, X=X)
    




###################### REHVI #############################
class qExpectedHypervolumeImprovement_FixedSizedParetoFront(
    MultiObjectiveMCAcquisitionFunction, SubsetIndexCachingMixin
):
    def __init__(
        self,
        model: Model,
        ref_point: Union[List[float], Tensor],
        partitioning: NondominatedPartitioning,
        M_size: int,
        pareto_front: Tensor,
        sampler: Optional[MCSampler] = None,
        objective: Optional[MCMultiOutputObjective] = None,
        constraints: Optional[List[Callable[[Tensor], Tensor]]] = None,
        X_pending: Optional[Tensor] = None,
        eta: Optional[Union[Tensor, float]] = 1e-3,
        fat: bool = False,

    ) -> None:
        
        self.M_size = M_size
        self.pareto_front = pareto_front

        assert pareto_front.shape[0]==M_size, "the pareto front should have M solutions"

        if len(ref_point) != partitioning.num_outcomes:
            raise ValueError(
                "The length of the reference point must match the number of outcomes. "
                f"Got ref_point with {len(ref_point)} elements, but expected "
                f"{partitioning.num_outcomes}."
            )
        ref_point = torch.as_tensor(
            ref_point,
            dtype=partitioning.pareto_Y.dtype,
            device=partitioning.pareto_Y.device,
        )
        super().__init__(
            model=model,
            sampler=sampler,
            objective=objective,
            constraints=constraints,
            eta=eta,
            X_pending=X_pending,
        )
        self.register_buffer("ref_point", ref_point)
        # cell_bounds = partitioning.get_hypercell_bounds()
        # self.register_buffer("cell_lower_bounds", cell_bounds[0])
        # self.register_buffer("cell_upper_bounds", cell_bounds[1])
        SubsetIndexCachingMixin.__init__(self)
        self.fat = fat


    def _compute_hv(self, samples: Tensor, X: Optional[Tensor] = None) -> Tensor:

        m = self.pareto_front.shape[0]

        assert m >= 1, "Pareto front must have at least 1 point."

        obj = self.objective(samples, X=X)
        q = obj.shape[-2]

        # print('q is ',q)
        input_dim = self.ref_point.shape[0]


        hv_computer_temp = Hypervolume(self.ref_point)
        hv_candidate_current_best = hv_computer_temp.compute(self.pareto_front)
        

        
        
        record_2 = torch.zeros(
                m,
                obj.shape[0],
                obj.shape[1], # this is the batch_shape
                dtype=obj.dtype,
                device=self.ref_point.device,)
        


        # print('obj',obj.shape)

        for iter in range(m):

            # Remove point i to create a candidate subset
            PF_candidate_subset =  torch.cat((self.pareto_front[:iter], self.pareto_front[iter+1:]), dim=0)
            # print('check temp pf: ',PF_candidate_subset==choice_M[0])


            hv_computer_temp = Hypervolume(self.ref_point)
            hv_candidate_subset = hv_computer_temp.compute(PF_candidate_subset)


            partitioning = FastNondominatedPartitioning(ref_point=self.ref_point,Y=PF_candidate_subset.reshape(self.M_size-1,input_dim))
            cell_bounds = partitioning.get_hypercell_bounds()
            self.register_buffer("cell_lower_bounds", cell_bounds[0])
            self.register_buffer("cell_upper_bounds", cell_bounds[1])



            if self.constraints is not None:
                feas_weights = compute_smoothed_feasibility_indicator(
                    constraints=self.constraints,
                    samples=samples,
                    eta=self.eta,
                    fat=self.fat,
                )  # `sample_shape x batch-shape x q`
                
            device = self.ref_point.device
            q_subset_indices = self.compute_q_subset_indices(q_out=q, device=device)
            batch_shape = obj.shape[:-2]
            # this is n_samples x input_batch_shape x
            areas_per_segment = torch.zeros(
                *batch_shape,
                self.cell_lower_bounds.shape[-2],
                dtype=obj.dtype,
                device=device,
            )
            cell_batch_ndim = self.cell_lower_bounds.ndim - 2
            sample_batch_view_shape = torch.Size(
                [
                    batch_shape[0] if cell_batch_ndim > 0 else 1,
                    *[1 for _ in range(len(batch_shape) - max(cell_batch_ndim, 1))],
                    *self.cell_lower_bounds.shape[1:-2],
                ]
            )
            view_shape = (
                *sample_batch_view_shape,
                self.cell_upper_bounds.shape[-2],
                1,
                self.cell_upper_bounds.shape[-1],
            )
            for i in range(1, self.q_out + 1):
                # TODO: we could use batches to compute (q choose i) and (q choose q-i)
                # simultaneously since subsets of size i and q-i have the same number of
                # elements. This would decrease the number of iterations, but increase
                # memory usage.
                q_choose_i = q_subset_indices[f"q_choose_{i}"]
                # this tensor is mc_samples x batch_shape x i x q_choose_i x m
                obj_subsets = obj.index_select(dim=-2, index=q_choose_i.view(-1))
                obj_subsets = obj_subsets.view(
                    obj.shape[:-2] + q_choose_i.shape + obj.shape[-1:]
                )
                # since all hyperrectangles share one vertex, the opposite vertex of the
                # overlap is given by the component-wise minimum.
                # take the minimum in each subset
                overlap_vertices = obj_subsets.min(dim=-2).values
                # add batch-dim to compute area for each segment (pseudo-pareto-vertex)
                # this tensor is mc_samples x batch_shape x num_cells x q_choose_i x m
                overlap_vertices = torch.min(
                    overlap_vertices.unsqueeze(-3), self.cell_upper_bounds.view(view_shape)
                )
                # subtract cell lower bounds, clamp min at zero
                lengths_i = (
                    overlap_vertices - self.cell_lower_bounds.view(view_shape)
                ).clamp_min(0.0)
                # take product over hyperrectangle side lengths to compute area
                # sum over all subsets of size i
                areas_i = lengths_i.prod(dim=-1)
                # if constraints are present, apply a differentiable approximation of
                # the indicator function
                if self.constraints is not None:
                    feas_subsets = feas_weights.index_select(
                        dim=-1, index=q_choose_i.view(-1)
                    ).view(feas_weights.shape[:-1] + q_choose_i.shape)
                    areas_i = areas_i * feas_subsets.unsqueeze(-3).prod(dim=-1)
                areas_i = areas_i.sum(dim=-1)
                # Using the inclusion-exclusion principle, set the sign to be positive
                # for subsets of odd sizes and negative for subsets of even size
                areas_per_segment += (-1) ** (i + 1) * areas_i

                
            record_2[iter,:] = (areas_per_segment.sum(dim=-1)+hv_candidate_subset-hv_candidate_current_best).clamp_min(0.0)
            
        max_values, _ = torch.max(record_2, dim=0)  # Get max along the first dimension (M) 
        max_mean = max_values.mean(dim=0)


        return max_mean 



    @concatenate_pending_points
    @t_batch_mode_transform()
    def forward(self, X: Tensor) -> Tensor:
        posterior = self.model.posterior(X)
        samples = self.get_posterior_samples(posterior)

        
        return self._compute_hv(samples=samples, X=X)



class AcqFuncWrapper:
    def __init__(self, acq_func):
        self.acq_func = acq_func
        self.eval_count = 0

    def __call__(self, X):
        self.eval_count += X.shape[0] if hasattr(X, "shape") else 1
        return self.acq_func(X)

    # Forward other needed attributes/methods
    def __getattr__(self, name):
        return getattr(self.acq_func, name)




def optimize_qehvi_fixed_size_PF(type, model,M, pareto_front,pareto_set, train_x, train_obj, sampler,ref_point,problem,batch_size=BATCH_SIZE,**kwargs):
    '''
    When M = 0, pareto_front is the full Pareto front; When M>0, pareto_front is the Pareto front of size M.
    '''


    standard_bounds = torch.zeros(2, problem.dim)
    standard_bounds[1] = 1

    mean = kwargs.get('mean', False)

    log = kwargs.get('log', False)

    if type == 'EHVI':
        assert M == 0, "M should be 0 for no_size method"
        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=pareto_front,
        )
        acq_func = qExpectedHypervolumeImprovement(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,
        )


        batch_size_temp = batch_size

        # acq_func = AcqFuncWrapper(acq_func)

    elif type == 'UCB':
        assert M == 0, "M should be 0 for no_size method"

        partitioning = FastNondominatedPartitioning(
            ref_point=ref_point,
            Y=pareto_front,
        )

        acq_func = UCB_HV(
            model=model,
            ref_point=ref_point,
            partitioning=partitioning,
            sampler=sampler,)


        batch_size_temp = batch_size


    elif type == 'EHVI_M':

        assert pareto_front.shape[0] == M, "the pareto front size is not M"

        partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=pareto_front.reshape(M,-1))
        acq_func = qExpectedHypervolumeImprovement_FixedSizedParetoFront(
                                                model=model,
                                                ref_point=ref_point,
                                                partitioning=partitioning,
                                                M_size = M,
                                                pareto_front=pareto_front,
                                                sampler=sampler,
                                                )
        
        batch_size_temp = batch_size

        # acq_func = AcqFuncWrapper(acq_func)
   
    
    elif type == 'HD_EI':
        assert pareto_front.shape[0] == M, "the pareto front size is not M"


        mean = kwargs.get('mean', False)

        partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=ref_point.reshape(1,-1))

        if not mean:
            acq_func = qExpectedHypervolumeImprovement_FixedSizedParetoFront_full(
                                                        model=model,
                                                        ref_point=ref_point,
                                                        partitioning=partitioning,
                                                        pareto_front=pareto_front,  # this pareto pareto should be the best M subset pareto front
                                                        sampler=sampler,
                                                        
                                                    )
        if mean:
            acq_func = qExpectedHypervolumeImprovement_FixedSizedParetoFront_full(
                                                        model=model,
                                                        ref_point=ref_point,
                                                        partitioning=partitioning,
                                                        pareto_front=pareto_front,
                                                        sampler=sampler,
                                                        mean = True
                                                    )

        batch_size_temp = M*batch_size

        # acq_func = AcqFuncWrapper(acq_func)


    elif type == 'HD_logEI':

        log = kwargs.get('log', False)

        assert pareto_front.shape[0] == M, "the pareto front size is not M"


        partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=ref_point.reshape(1,-1))

        if not mean:
            acq_func = qExpectedHypervolumeImprovement_FixedSizedParetoFront_full(
                                                        model=model,
                                                        ref_point=ref_point,
                                                        partitioning=partitioning,
                                                        pareto_front=pareto_front,  # this pareto pareto should be the best M subset pareto front
                                                        sampler=sampler,
                                                        log=log
                                                        
                                                    )
        if mean:
            acq_func = qExpectedHypervolumeImprovement_FixedSizedParetoFront_full(
                                                        model=model,
                                                        ref_point=ref_point,
                                                        partitioning=partitioning,
                                                        pareto_front=pareto_front,
                                                        sampler=sampler,
                                                        mean = True
                                                        
                                                    )



        batch_size_temp = M*batch_size

        # acq_func = AcqFuncWrapper(acq_func)


    elif type == 'HD_UCB':
        assert pareto_front.shape[0] == M, "the pareto front size is not M"

        beta = kwargs.get('beta', False)

        partitioning = FastNondominatedPartitioning(ref_point=ref_point,Y=ref_point.reshape(1,-1))

        acq_func = UCB_HypervolumeImprovement(beta=beta,
                                            model=model,
                                            ref_point=ref_point,
                                            partitioning=partitioning,
                                            sampler=sampler,)
        
        batch_size_temp = M*batch_size
        # acq_func = AcqFuncWrapper(acq_func)


    candidates, val = optimize_acqf(
                                        acq_function=acq_func,
                                        bounds=standard_bounds,
                                        q=batch_size_temp,
                                        num_restarts=NUM_RESTARTS,
                                        raw_samples=RAW_SAMPLES,  
                                        options=OPTIONS,
                                        sequential=False,)
        


    

    return candidates, val #, acq_func.eval_count





def get_reference_chebyshev_scalarization(
    weights: Tensor, 
    Y: Tensor, 
    reference_point: Tensor,
    alpha: float = 0.05
) -> Callable[[Tensor, Union[Tensor, None]], Tensor]:
    r"""Construct an augmented Chebyshev scalarization with reference point.
    
    The reference point augmented Chebyshev scalarization is given by
    g(y) = max_i(w_i * (ref_i - y_i)) + alpha * sum_i(w_i * (ref_i - y_i))
    where the goal is to minimize g(y) and get as close as possible to the reference point.
    
    Since the default in BoTorch is to maximize all objectives, this method constructs 
    a Chebyshev scalarization where the inputs are first multiplied by -1, so that all 
    objectives are to be minimized. Then, it computes g(y) (which should be minimized), 
    and returns -g(y), which should be maximized.
    
    Args:
        weights: A `m`-dim tensor of weights. Positive for maximization and negative 
                for minimization.
        Y: A `n x m`-dim tensor of observed outcomes, which are used for scaling 
           the outcomes to [0,1] or [-1,0]. If `n=0`, then outcomes are left unnormalized.
        reference_point: A `m`-dim tensor specifying the reference point for each objective.
                        Should be in the same space as Y (before normalization).
        alpha: Parameter governing the influence of the weighted sum term. 
               The default value comes from [Knowles2005]_.
    
    Returns:
        Transform function using the objective weights and reference point.
        
    Example:
        >>> weights = torch.tensor([0.75, 0.25])
        >>> Y = torch.tensor([[1.0, 2.0], [3.0, 1.0], [2.0, 3.0]])
        >>> ref_point = torch.tensor([0.5, 0.5])  # Reference point to aim for
        >>> transform = get_reference_chebyshev_scalarization(weights, Y, ref_point)
    """
    
    # Multiply Y by -1 since chebyshev_obj assumes all objectives should be minimized
    Y = -Y
    ref = -reference_point  # Convert reference point to same space
    
    if weights.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "weights must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {weights.shape} and {Y.shape}."
        )
    elif Y.ndim > 2:
        raise NotImplementedError("Batched Y is not currently supported.")
        
    if reference_point.shape != Y.shape[-1:]:
        raise BotorchTensorDimensionError(
            "reference_point must be an `m`-dim tensor where Y is `... x m`."
            f"Got shapes {reference_point.shape} and {Y.shape}."
        )

    def chebyshev_obj(Y: Tensor, ref_normalized: Tensor, X: Tensor = None) -> Tensor:
        # Distance from reference point
        diff = Y - ref_normalized
        product = weights * diff
        return product.max(dim=-1).values + alpha * product.sum(dim=-1)

    # A boolean mask indicating if minimizing an objective
    minimize = weights < 0
    
    if Y.shape[-2] == 0:
        if minimize.any():
            raise UnsupportedError(
                "negative weights (for minimization) are only supported if "
                "Y is provided."
            )
        # If there are no observations, we do not need to normalize the objectives
        def obj(Y: Tensor, X: Tensor = None) -> Tensor:
            # Convert Y to minimization space and use unnormalized reference point
            Y_min = -Y
            ref_min = -reference_point
            
            # Distance from reference point
            diff = ref_min - Y_min
            product = weights * diff
            chebyshev_val = product.max(dim=-1).values + alpha * product.sum(dim=-1)
            
            # multiply the scalarization by -1, so that the scalarization should
            # be maximized
            return -chebyshev_val
        
        return obj

    # Set the bounds to be [min(Y_m), max(Y_m)], for each objective m.
    Y_bounds = torch.stack([Y.min(dim=-2).values, Y.max(dim=-2).values])
    
    ref_normalized = normalize(ref, bounds=Y_bounds)

    
    # If minimizing an objective, convert ref_normalized values to [-1,0], such that max(w*(ref-y)) makes sense when all w*(ref-y)'s should be positive
    ref_normalized = ref_normalized.clone()  # Make a copy to avoid in-place modification
    ref_normalized[minimize] = ref_normalized[minimize] - 1

    def obj(Y: Tensor, X: Tensor = None) -> Tensor:
        # scale to [0,1]
        Y_normalized = normalize(-Y, bounds=Y_bounds)
        
        # If minimizing an objective, convert Y_normalized values to [-1,0], such that max(w*(ref-y)) makes sense, we want all w*(ref-y)'s to be positive
        Y_normalized = Y_normalized.clone()  # Make a copy to avoid in-place modification
        Y_normalized[..., minimize] = Y_normalized[..., minimize] - 1
        
        # multiply the scalarization by -1, so that the scalarization should be maximized
        return -chebyshev_obj(Y=Y_normalized, ref_normalized=ref_normalized)

    return obj





