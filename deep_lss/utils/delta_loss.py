# Copyright (C) 2022 ETH Zurich, Institute for Particle Physics and Astrophysics

"""
Created December 2022
Author: Arne Thomsen, Janis Fluri

Adapted from
https://cosmo-gitlab.phys.ethz.ch/jafluri/cosmogrid_kids1000/-/blob/master/kids1000_analysis/losses.py
by Janis Fluri, 
the main difference is that this version uses PyTorch tensor operations.
"""

import numpy as np
import torch

from deep_lss.utils import summary

from msfm.utils import logger

LOGGER = logger.get_logger(__file__)


def torch_matrix_condition(m):
    """Calculate the matrix condition number of an input m over the last two axis, defined as the ratio of the largest
    and smallest singular value

    Args:
        m (torch.Tensor): The input tensor of shape [...,N,M]

    Returns:
        torch.Tensor: The condition number of shape [...]
    """
    s = torch.linalg.svdvals(m)
    return s[..., 0] / s[..., -1].clamp_min(torch.finfo(m.dtype).eps)


def get_jac_and_cov_matrix(
    predictions, n_params, n_same, off_sets, n_output=None, summary_writer=None, training=False, strategy=None
):
    """Calculates the covariance of the fiducial predictions and the jacobians of the means and returns it. It assumes
    a specific ordering of the predictions.

    Args:
        predictions (torch.Tensor): Predictions of shape (n_same * (1 + 2 * n_params), n_params) in a fixed ordering.
        n_params (int): Number of underlying model parameters.
        n_same (int): Number of realizations of the same parameter (the perturbations don't count).
        off_sets (np.ndarray): The finite differences in the underlying parameters to calculate the Jacobian.
        n_output (_type_, optional): dimensionality of the summary statistic, defaults to predictions.shape[-1] if None.
        summary_writer (optional): Used to write tensorboard summaries. Defaults to None.
        training (bool, optional): Wheter the network is currently training. If False, no summary is written even if a
            writer is provided. Defaults to False.
            strategy (optional): Reserved for distributed loss aggregation. Defaults to None, then training is local.

    Returns:
        torch.Tensor: Covariances and Jacobians, these have shape (n_output/n_params, n_output, n_output), where n_output
            is the dimensionality of the summary statistic.
    """
    # needs to be a torch.Tensor, has shape (n_same * (1 + 2 * n_params), n_params)
    if isinstance(predictions, np.ndarray):
        predictions = torch.as_tensor(predictions)

    # get number of outputs
    if n_output is None:
        n_output = int(predictions.shape[-1])

    # split the local output, len(splits) = (1 + 2 * n_params), split.shape = splits[0].shape = (?, n_same, n_output)
    splits = [
        torch.reshape(split, shape=(-1, n_same, n_output))
        for split in torch.chunk(predictions, chunks=2 * n_params + 1, dim=0)
    ]

    if strategy is not None:
        raise NotImplementedError("Distributed delta-loss aggregation must be provided by the PyTorch training loop")
    # minus one because this is the sample covariance
    cov_normalization = predictions.new_tensor(float(n_same) - 1.0)

    # summary
    if training:
        # len(param_splits) = n_params, param_splits[0].shape = (?, n_same, n_output)
        param_splits = torch.split(splits[0], split_size_or_sections=1, dim=-1)

        for num, single_param in enumerate(param_splits):
            summary.write_summary(
                f"delta_param_{num}_hist", single_param, summary_writer, training, summary_type="histogram"
            )

    # get the covariance NOTE the mean is taken over the n_same, the (local/global) batch size
    mean = torch.mean(splits[0], dim=1, keepdim=True)

    # shape (n_output/n_params, n_same, n_params)
    outmm = torch.subtract(splits[0], mean)

    # shape (n_output/n_params, n_output, n_output)
    cov = torch.divide(torch.einsum("hjk,hjl->hkl", outmm, outmm), cov_normalization)

    # handle off sets and renormalization
    off_sets = torch.as_tensor(off_sets, dtype=predictions.dtype, device=predictions.device)

    # get mean derivatives NOTE the mean is taken over the n_same, the batch size
    derivatives = []
    for i in range(n_params):
        mean_minus = torch.mean(splits[2 * (i + 1) - 1], dim=1, keepdim=False)
        mean_plus = torch.mean(splits[2 * (i + 1)], dim=1, keepdim=False)
        derivatives.append(torch.divide(torch.subtract(mean_plus, mean_minus), 2.0 * off_sets[i]))

    # stack the derivatives to form the Jacobian, shape (n_output/n_params, n_output, n_output)
    jacobian = torch.stack(derivatives, dim=-1)

    return cov, jacobian


# def get_fisher_from_cov_jacobian(cov, jacobian):
#     """Calculates the approximate fisher information given a covariance matrix and jacobian

#     Args:
#         cov (torch.Tensor): The covariance matrix of the summary
#         jacobian (torch.Tensor): The jacobian of the summary

#     Returns:
#         torch.Tensor: The approximate fisher matrix
#     """

#     # calculate approximate fisher information like below eq. (14) in https://arxiv.org/pdf/2107.09002.pdf
#     # F = inv(J^-1 cov J^T^-1) = J^T cov^-1 J
#     inv_cov = torch.linalg.inv(cov)
#     fisher = torch.einsum("aij,ajk->aik", inv_cov, jacobian)
#     fisher = torch.einsum("aji,ajk->aik", jacobian, fisher)

#     return fisher


def delta_loss(
    predictions,
    n_params,
    n_same,
    off_sets,
    # regularization
    force_params_value=0.0,
    force_params_weight=1.0,
    jac_weight=100.0,
    cov_loss=False,
    jac_cond_weight=None,
    # summary statistic
    n_output=None,
    n_partial=None,
    weights=None,
    no_correlations=False,
    # numerical stability
    use_log_det=True,
    tikhonov_regu=False,
    eps=1e-32,
    # summary
    summary_writer=None,
    training=True,
    img_summary=False,
    print_scalar=False,
    summary_suffix="",
    # distribution
    strategy=None,
):
    """This function calculates the delta loss which tries to maximize the information of the summary statistics. Note
    it needs the predictions to be ordered in a specific way:
        * The shape of the predictions is (n_points * n_same * (2 * n_params + 1), n_output)
        * If one splits the predictions into (2 * n_params + 1) parts among the first axis one has the following scheme:
            * The first part was generated with the unperturbed parameters
            * The second part was generated with parameters where off_sets[0] was subtracted from the first param
            * The third part was generated with parameters where off_sets[0] was added from to first param
            * The fourth part was generated with parameters where off_sets[1] was subtracted from the second param
            * and so on

    Args:
        predictions (torch.Tensor): The predictions a.k.a. summary statistics in the specified ordering.
        n_params (int): Number of underlying (cosmological) model parameters.
        n_same (int): Number of (uperturbed) summaries coming from the same parameter set, this is the same as the
            (local) batch size
        off_sets (np.ndarray): The off-sets used to perturb the original (fiducial) parameters. These are used as the
            finite differences in the computation of the Jacobian.
        force_params_value (float, np.ndarray, optional): Either None or a set of parameters with shape
            (n_points, 1, n_output) which is used to compute a square loss of the unperturbed summaries. It is useful
            to set this for example to zeros such that the network does not produces arbitrary high summary values.
            Defaults to None, float inputs are broadcast to the appropriate shape.
        force_params_weight (float, optional): The weight of the square loss of force_params_value. Defaults to 1.0.
        jac_weight (float, optional): The weight of the Jacobian loss, which forces the Jacobian of the summaries
            to be close to unity (or identity matrix). Defaults to 100.0.
        cov_loss (bool, optional): If true, the jac_weight will be used as cov_weight, i.e. loss cov mat will be
            forced to be close to the identity matrix. Note that there will be an additional term forcing the inverse
            of the covariance to be close to the identity as well since the cov is guaranteed to be square. This is the
            same as Luca's regularization term, but without the adaptive weight. Defaults to False.
        jac_cond_weight (float, optional): If not None, this weight is used to add an additional loss using the matrix
            condition number of the jacobian. Defaults to None.
        n_output (int, optional): Dimensionality of the summary statistic. Defaults to None, which corresponds to
            predictions.shape[-1].
        n_partial (np.ndarray, optional): To train only on a subset of parameters and not all underlying model
            parameter. Defaults to None which means the information inequality is minimized in a normal fashion. Note
            that due to the necessity of some algebraic manipulations n_partial == None and n_partial == n_params lead
            to slightly different behaviour. Defaults to None.
        weights (np.ndarray, optional): An 1d array of length n_points, used as weights in means of the different
            points. Defaults to None.
        no_correlations (bool, optional): Do not consider correlations between the parameter, this means that one tries
            to find an optimal summary (single value) for each underlying model parameter, only possible if
            n_output == n_params. Defaults to False.
        use_log_det (bool, optional): Use the log of the determinants in the information inequality, should be True. If
            False the information inequality is not minimized in a proper manner and the training can become unstable.
            Defaults to True.
        tikhonov_regu (bool, optional): Use Tikhonov regularization of matrices e.g. to avoid vanishing determinants.
            This is the recommended regularization method as it allows the usage of some optimized routines. Defaults
            to False.
        eps (float, optional): A small positive value used for regularization of things like logs etc. This should
            only be increased if tikhonov_regu is used and a error is raised. Defaults to 1e-32.
        summary_writer (optional): The writer used to write tensorboard summaries. Defaults
            to None.
        training (bool, optional): Whether the loss is used for training. If False, no summaries will be written even
            if a summary_writer is supplied. Defaults to True.
        img_summary (bool, optional): Save image summaries of the Jacobian and the covariance. Defaults to False.
        print_scalar (bool, optional): Print the scalar value of the loss to the console. Defaults to False.
        summary_suffix (str, optional): A label used to identify the summaries in tensorboard. Defaults to "".
        strategy (optional): Reserved for distributed loss aggregation.

    Raises:
        ValueError: When there are specifications that conflict with the no_correlations boolean.

    Returns:
        torch.Tensor: The loss value, which can be negative.
    """

    LOGGER.warning(f"Tracing delta_loss")

    # TODO: A fixed epsilon can lead to some problems. E.g. in tikonov regularization might fail because the lack
    # TODO: of precision. A possible solution would be to use the machine epsilon for added regularization
    # TODO: and a fixed epsilon for absolut regulatization (division or log errors...)

    # get cov and jac of shapes (n_output/n_params, n_output, n_output)
    cov, jacobian = get_jac_and_cov_matrix(
        predictions=predictions,
        n_params=n_params,
        n_same=n_same,
        off_sets=off_sets,
        n_output=n_output,
        summary_writer=summary_writer,
        training=training,
        # distribution
        strategy=strategy,
    )

    # nice output
    summary.write_summary(
        "delta_jacobian_hist" + summary_suffix, jacobian, summary_writer, training, summary_type="histogram"
    )
    if img_summary:
        jac_img = torch.unsqueeze(jacobian, dim=3)
        jac_max = torch.amax(jac_img, dim=(1, 2), keepdim=True)
        jac_min = torch.amin(jac_img, dim=(1, 2), keepdim=True)
        jac_img = torch.divide(jac_img - jac_min, jac_max - jac_min)
        summary.write_summary(
            "delta_jacobian_img" + summary_suffix, jac_img, summary_writer, training, summary_type="image"
        )

        cov_img = torch.unsqueeze(cov, dim=3)
        cov_max = torch.amax(cov_img, dim=(1, 2), keepdim=True)
        cov_min = torch.amin(cov_img, dim=(1, 2), keepdim=True)
        cov_img = torch.divide(cov_img - cov_min, cov_max - cov_min)
        summary.write_summary(
            "delta_covariance_img" + summary_suffix, cov_img, summary_writer, training, summary_type="image"
        )

        # get corrlation matrix
        v = torch.sqrt(torch.diagonal(cov, dim1=-2, dim2=-1))
        outer_v = torch.einsum("ai,aj->aij", v, v)
        cor = torch.divide(cov, outer_v)
        cor_img = torch.unsqueeze(cor, dim=3)
        # fit between 0 and 1
        cor_img = torch.add(0.5, 0.5 * cor_img)
        summary.write_summary(
            "delta_correlation_img" + summary_suffix, cor_img, summary_writer, training, summary_type="image"
        )

    # in case predictions is a numpy array
    if isinstance(predictions, np.ndarray):
        predictions = torch.as_tensor(predictions)

    # get number of outputs
    if n_output is None:
        n_output = int(predictions.shape[-1])

    # note worthy stuff
    if n_output > n_params and jac_weight > 0.0:
        LOGGER.warning(
            "The weight of the Jacobian loss should be zero, if you have a summary that has a higher"
            " dimension as the number of model params!"
        )

    if no_correlations and (n_output != n_params):
        raise ValueError("Independent summaries (no_correlations) is only possible if n_output == n_params")

    if no_correlations and n_partial is not None:
        raise ValueError("Independent summaries (no_correlations) is only possible if n_partial is None")

    if (force_params_weight is not None) and (force_params_weight < 0.0):
        raise ValueError(f"force_params_weight = {force_params_weight} must be positive")

    if (jac_weight is not None) and (jac_weight < 0.0):
        raise ValueError(f"jac_weight = {jac_weight} must be positive")

    if (jac_cond_weight is not None) and (jac_cond_weight < 0.0):
        raise ValueError(f"jac_cond_weight = {jac_cond_weight} must be positive")

    # main loss
    if use_log_det:
        # check if we are in no correlation regime
        if no_correlations:
            # tikhonov_regu and normal regu is the same in this case
            cov_diag = torch.diagonal(cov, dim1=-2, dim2=-1)
            cov_log_det = torch.log(cov_diag + predictions.new_tensor(eps))
            jac_diag = torch.diagonal(jacobian, dim1=-2, dim2=-1)
            jac_log_det = torch.log(torch.square(jac_diag) + predictions.new_tensor(eps))
            # the factor of 2 is in the square of the jac_diag
            cov_det_loss = torch.mean(torch.subtract(cov_log_det, jac_log_det))

        # NOTE use everything, this is the default branch
        elif n_partial is None:
            # torch.logdet is much better for the backprop, but fails if the det is zero
            # should we do cov + eps*identity?
            if tikhonov_regu:
                identity = torch.eye(n_params, dtype=predictions.dtype, device=predictions.device).unsqueeze(0) * predictions.new_tensor(eps)
                # we use that 2*log(det(A)) = log(det(A)^2) = log(det(A)*det(A)) = log(det(A)*det(A^T))
                #                           = log(det(A*A^T))identity
                jt_j = torch.einsum("aji,ajk->aik", jacobian, jacobian)
                jac_log_det = torch.logdet(torch.add(jt_j, identity))
                cov_log_det = torch.logdet(torch.add(cov, identity))
                cov_det_loss = torch.subtract(cov_log_det, jac_log_det)
            # NOTE no tikhonov regularization is the default
            else:
                jac_log_det = torch.log(torch.abs(torch.linalg.det(jacobian)) + predictions.new_tensor(eps))
                # We add a abs here because of instabilities
                cov_log_det = torch.log(torch.abs(torch.linalg.det(cov)) + predictions.new_tensor(eps))

                if print_scalar:
                    print(f"cov_log_det: {cov_log_det}")
                    print(f"jac_log_det: {-2.0 * jac_log_det}")

                cov_det_loss = torch.subtract(cov_log_det, 2.0 * jac_log_det)

        else:
            # we use only the first n_partial params
            j_part = jacobian[:, :, :n_partial]

            # now we need to calculate log(det(J^T cov J)) - log(det(J^T J))
            cov_j = torch.einsum("aij,ajk->aik", cov, j_part)
            jt_cov_j = torch.einsum("aji,ajk->aik", j_part, cov_j)
            jt_j = torch.einsum("aji,ajk->aik", j_part, j_part)

            if tikhonov_regu:
                id_dim = np.minimum(n_params, n_partial)
                identity = torch.eye(id_dim, dtype=predictions.dtype, device=predictions.device).unsqueeze(0) * predictions.new_tensor(eps)
                jac_log_det = torch.logdet(torch.add(jt_j, identity))
                cov_log_det = torch.logdet(torch.add(jt_cov_j, identity))
            else:
                # We add a abs here because of instabilities
                jac_log_det = torch.log(torch.abs(torch.linalg.det(jt_j)) + predictions.new_tensor(eps))
                cov_log_det = torch.log(torch.abs(torch.linalg.det(jt_cov_j)) + predictions.new_tensor(eps))

            cov_det_loss = torch.subtract(cov_log_det, 2.0 * jac_log_det)

    else:
        # dividing by the jac_det (for info inequality) does not work...
        LOGGER.warning(
            f"You are using use_log_det=False. Only the determinant of the covariance matrix will be"
            f" optimized. This loss might be unbouned and could lead to unstable training."
        )
        cov_det_loss = torch.linalg.det(cov)

    if weights is not None:
        # normalize the weights
        weights = torch.as_tensor(weights, dtype=predictions.dtype, device=predictions.device); weights = torch.divide(weights, torch.sum(weights).clamp_min(predictions.new_tensor(eps)))
        # do a weighted mean
        cov_det_loss = torch.multiply(weights, cov_det_loss)

    # normal mean, this is taken if the output dimension of the summary statistic is different than the number of
    # parameters. So nothing happens here if n_output = n_params, because then cov_det only has one entry.
    cov_det_loss = torch.mean(cov_det_loss)
    summary.write_summary(
        "loss/delta_cov_det" + summary_suffix, cov_det_loss, summary_writer, training, print_scalar=print_scalar
    )

    loss = cov_det_loss

    # jacobian loss (log of this is unstable)
    if jac_weight is not None:
        if cov_loss:
            jac_label = "loss/delta_covariance"

            diff = torch.subtract(cov, torch.eye(n_output, n_output, dtype=predictions.dtype, device=predictions.device).unsqueeze(0))
            jac_loss = torch.mean(torch.square(diff), dim=(1, 2))

            # add loss to the inverse
            diff = torch.subtract(
                torch.linalg.inv(cov), torch.eye(n_output, n_output, dtype=predictions.dtype, device=predictions.device).unsqueeze(0)
            )
            jac_loss += torch.mean(torch.square(diff), dim=(1, 2))
            jac_loss *= 0.5

        # NOTE this is the default branch
        else:
            jac_label = "loss/delta_jacobian"

            # shape (n_output/n_params, n_output, n_output)
            diff = torch.subtract(jacobian, torch.eye(n_output, n_params, dtype=predictions.dtype, device=predictions.device).unsqueeze(0))
            if n_partial is None:
                # use everything
                jac_loss = torch.mean(torch.square(diff), dim=(1, 2))
            else:
                # only n_part
                jac_loss = torch.mean(torch.square(diff)[:, :, :n_partial], dim=(1, 2))

        if weights is not None:
            jac_loss = torch.multiply(weights, jac_loss)

        jac_loss = torch.mean(jac_loss)
        jac_loss = jac_weight * jac_loss

        summary.write_summary(
            jac_label + summary_suffix, jac_loss, summary_writer, training, print_scalar=print_scalar
        )
        loss = torch.add(loss, jac_loss)

    # condition number loss
    if jac_cond_weight is not None:
        if n_partial is not None:
            c = torch_matrix_condition(jacobian[..., :n_partial])

        else:
            c = torch_matrix_condition(jacobian)

        if weights is not None:
            c = torch.multiply(weights, c)

        jac_cond_loss = torch.mean(c)
        jac_cond_loss = jac_cond_weight * jac_cond_loss

        summary.write_summary(
            "loss/delta_jacobian_cond" + summary_suffix,
            jac_cond_loss,
            summary_writer,
            training,
            print_scalar=print_scalar,
        )
        loss = torch.add(loss, jac_cond_loss)

    # diff loss
    if (force_params_value is not None) and (force_params_weight is not None):
        # calculate square distance between fidu mean and preds
        mid_params = torch.chunk(predictions, chunks=2 * n_params + 1, dim=0)[0]

        # reshape
        mid_params = torch.reshape(mid_params, shape=(-1, n_same, n_output))

        # penalty
        diff = torch.subtract(mid_params, force_params_value)
        diff_loss = torch.square(torch.mean(diff, dim=1))

        if weights is not None:
            # reduce mean over the last axis (n params)
            diff_loss = torch.mean(diff_loss, dim=1)
            # weight and mean
            diff_loss = torch.multiply(diff_loss, weights)

        # simple mean reduction
        diff_loss = torch.mean(diff_loss)

        # force weight
        diff_loss = force_params_weight * diff_loss

        summary.write_summary(
            "loss/delta_diff" + summary_suffix, diff_loss, summary_writer, training, print_scalar=print_scalar
        )
        loss = torch.add(loss, diff_loss)

    return loss
