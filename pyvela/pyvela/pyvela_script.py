"""Script for running pulsar timing & noise analysis using Vela.jl with emcee."""

from copy import deepcopy
import json
import os
import shutil
import warnings
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

import emcee
import numpy as np
import astropy.units as u
from scipy.linalg import cholesky, cho_solve, solve_triangular, LinAlgError
from pint.residuals import Residuals, WidebandTOAResiduals

from pyvela import SPNTA
from pyvela.parameters import (
    analytic_marginalizable_names,
    analytic_marginalizable_prefixes,
)
from pyvela.results import SPNTAResults


def parse_args(argv):
    parser = ArgumentParser(
        prog="pyvela",
        description="A command line interface for the Vela.jl pulsar timing &"
        " noise analysis package. Supports emcee and PTMCMCSampler sampling. "
        "This may not be appropriate for more complex datasets. Write your own "
        "scripts for such cases.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "par_file",
        help="The pulsar ephemeris file. Should be readable using PINT. The "
        "uncertainties listed in the file will be used for 'cheat' priors where applicable.",
    )
    parser.add_argument(
        "tim_file",
        help="The pulsar TOA file. Should be readable using PINT. Either this or a JLSO file (-J) should be provided.",
    )
    parser.add_argument(
        "-J",
        "--jlso_file",
        help="The JLSO file containing pulsar timing and noise model & TOAs created using "
        "`pyvela-jlso`. JLSO files may need to be recreated after updating `Vela.jl` since "
        "the data format may change. These files are faster to read and parse.",
    )
    parser.add_argument(
        "-P",
        "--prior_file",
        help="A JSON file containing the prior distributions for each free parameter. (Ignored if `-J` option is used.)",
    )
    parser.add_argument(
        "--no_marg_gp_noise",
        action="store_true",
        help="Don't analytically marginalize the correlated Gaussian noise amplitudes.",
    )
    parser.add_argument(
        "-A",
        "--analytic_marg",
        nargs="+",
        default=[],
        help="Parameters to analytically marginalze (only some parameters are allowed).",
    )
    parser.add_argument(
        "-T",
        "--truth",
        help="Pulsar ephemeris file containing the true timing and noise parameter values. "
        "Relevant for simulation studies.",
    )
    parser.add_argument(
        "-C",
        "--cheat_prior_scale",
        default=100,
        type=float,
        help="The scale factor by which the frequentist uncertainties are multiplied to "
        "get the 'cheat' prior distributions.",
    )
    parser.add_argument(
        "-o",
        "--outdir",
        default="pyvela_results",
        help="The output directory. Will throw an error if it already exists (unless -f is given).",
    )
    parser.add_argument(
        "-f",
        "--force_rewrite",
        action="store_true",
        help="Force rewrite the output directory if it exists.",
    )
    parser.add_argument(
        "--sampler",
        choices=["emcee", "ptmcmc"],
        default="emcee",
        help="Sampler backend to use.",
    )
    parser.add_argument(
        "-N",
        "--nsteps",
        default=6000,
        type=int,
        help="Number of sampler iterations",
    )
    parser.add_argument(
        "-w",
        "--walkers",
        default=5,
        type=int,
        help="Number of ensemble MCMC walkers as a multiple of the number of dimensions",
    )
    parser.add_argument(
        "-b",
        "--burnin",
        default=1500,
        type=int,
        help="Burn-in length for MCMC chains",
    )
    parser.add_argument(
        "-t",
        "--thin",
        default=100,
        type=int,
        help="Thinning factor for MCMC chains",
    )
    parser.add_argument(
        "-r",
        "--resume",
        default=False,
        action="store_true",
        help="Resume from an existing run",
    )
    parser.add_argument(
        "-s",
        "--initial_sample_spread",
        default=0.3,
        type=float,
        help="Spread of the starting samples around the default parameter values. "
        "Must be > 0 and <= 1. 0 represents no spread and 1 represents prior draws.",
    )
    parser.add_argument(
        "-c",
        "--center_epochs",
        default=False,
        action="store_true",
        help="Center the epochs of the pulsar timing model.",
    )

    return parser.parse_args(argv)


def validate_input(args):
    assert os.path.isfile(
        args.par_file
    ), f"Invalid par file {args.par_file}. Make sure that the path is correct."
    assert os.path.isfile(
        args.tim_file
    ), f"Invalid tim file {args.tim_file}. Make sure that the path is correct."

    if args.jlso_file is not None:
        assert os.path.isfile(
            args.jlso_file
        ), f"Invalid JLSO file {args.jlso_file}. Make sure that the path is correct."

    assert args.prior_file is None or os.path.isfile(
        args.prior_file
    ), f"Prior file {args.prior_file} not found. Make sure the path is correct."
    assert args.truth is None or os.path.isfile(
        args.truth
    ), f"Truth par file {args.truth} not found.  Make sure the path is correct."

    if args.resume:
        args.force_rewrite = True
    assert args.force_rewrite or not os.path.isdir(
        args.outdir
    ), f"The output directory {args.outdir} already exists! Use `-f` option to force overwrite."

    assert (
        args.initial_sample_spread > 0 and args.initial_sample_spread <= 1
    ), "initial_sample_spread must be > 0 and <= 1."
    if args.sampler == "ptmcmc":
        assert not args.resume, "PTMCMCSampler resume is not supported yet."


def prepare_outdir(args):
    if args.force_rewrite and os.path.isdir(args.outdir) and not args.resume:
        shutil.rmtree(args.outdir)

    if not args.resume and not os.path.isdir(args.outdir):
        os.mkdir(args.outdir)

    if not os.path.exists(f"{args.outdir}/{os.path.basename(args.par_file)}"):
        shutil.copy(args.par_file, args.outdir)

    if not os.path.exists(f"{args.outdir}/{os.path.basename(args.tim_file)}"):
        shutil.copy(args.tim_file, args.outdir)

    if args.jlso_file is not None and not os.path.exists(
        f"{args.outdir}/{os.path.basename(args.jlso_file)}"
    ):
        shutil.copy(args.jlso_file, args.outdir)

    if args.prior_file is not None and not os.path.exists(
        f"{args.outdir}/{os.path.basename(args.prior_file)}"
    ):
        shutil.copy(args.prior_file, args.outdir)

    if args.truth is not None and not os.path.exists(
        f"{args.outdir}/{os.path.basename(args.truth)}"
    ):
        shutil.copy(args.truth, args.outdir)


def main(argv=None):
    args = parse_args(argv)

    if args.resume:
        # copy info from the prior run into the current arguments
        # to make sure they agree

        results = SPNTAResults(args.outdir)

        summary_info = results.summary
        args.par_file = results.input_par_file
        args.tim_file = results.input_tim_file
        args.cheat_prior_scale = summary_info["input"]["cheat_prior_scale"]
        args.analytic_marg = summary_info["input"]["analytic_marginalized_params"]
        args.prior_file = (
            f'{args.outdir}/{summary_info["input"]["custom_prior_file"]}'
            if summary_info["input"]["custom_prior_file"] is not None
            else None
        )
        args.jlso_file = results.jlso_file
        args.center_epochs = summary_info["input"]["center_epochs"]
        args.sampler = summary_info["sampler"]["sampler"]

    if "all" in args.analytic_marg:
        assert (
            len(args.analytic_marg) == 1
        ), "Other parameters cannot be specified when `-A all` is given."
        args.analytic_marg = (
            analytic_marginalizable_names + analytic_marginalizable_prefixes
        )

    validate_input(args)

    prepare_outdir(args)

    spnta = (
        SPNTA(
            args.par_file,
            args.tim_file,
            cheat_prior_scale=args.cheat_prior_scale,
            custom_priors=(args.prior_file if args.prior_file is not None else {}),
            marginalize_gp_noise=(not args.no_marg_gp_noise),
            analytic_marginalized_params=args.analytic_marg,
            center_epochs=args.center_epochs,
        )
        if args.jlso_file is None
        else SPNTA.load_jlso(
            args.jlso_file,
            args.par_file,
            args.tim_file,
            custom_prior_file=args.prior_file,
            cheat_prior_scale=args.cheat_prior_scale,
            analytic_marginalized_params=args.analytic_marg,
            center_epochs=args.center_epochs,
        )
    )

    if not args.resume and spnta.jlsofile is None:
        jlsofile = f"{args.outdir}/_{spnta.model.pulsar_name}.jlso"
        spnta.save_jlso(jlsofile)
        spnta.jlsofile = jlsofile

    if args.sampler == "emcee":
        samples_raw, sampler_info = run_emcee(spnta, args)
    elif args.sampler == "ptmcmc":
        samples_raw, sampler_info = run_ptmcmc(spnta, args)
    else:  # pragma: no cover
        raise ValueError(f"Unsupported sampler backend {args.sampler}")

    spnta.save_pre_analysis_summary(
        args.outdir,
        sampler_info,
        args.truth,
    )

    spnta.save_results(
        args.outdir,
        samples_raw,
        compute_autocorr=(args.sampler == "emcee"),
    )


def run_emcee(spnta: SPNTA, args):
    nwalkers = spnta.ndim * args.walkers

    p0 = get_start_samples(spnta, args.initial_sample_spread, nwalkers)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        spnta.ndim,
        spnta.lnpost_vectorized,
        moves=[emcee.moves.StretchMove(), emcee.moves.DESnookerMove()],
        vectorize=True,
        backend=emcee.backends.HDFBackend(f"{args.outdir}/chain.h5"),
    )
    if not args.resume:
        sampler.run_mcmc(
            p0, args.nsteps, progress=True, progress_kwargs={"mininterval": 1}
        )
    else:
        sampler.run_mcmc(
            None, args.nsteps, progress=True, progress_kwargs={"mininterval": 1}
        )

    samples_raw = sampler.get_chain(flat=True, discard=args.burnin, thin=args.thin)
    sampler_info = {
        "sampler": "emcee",
        "nwalkers": nwalkers,
        "nsteps": args.nsteps,
        "burnin": args.burnin,
        "thin": args.thin,
        "vectorized": True,
    }
    return samples_raw, sampler_info


def run_ptmcmc(spnta: SPNTA, args):
    """Run NANOGrav PTMCMCSampler using internal coordinates."""
    if args.walkers != 5:
        warnings.warn(
            "The --walkers option is emcee-specific and ignored when --sampler ptmcmc is used."
        )

    try:
        from PTMCMCSampler.PTMCMCSampler import PTSampler
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "PTMCMCSampler is required for --sampler ptmcmc. Install the PTMCMCSampler package."
        ) from e

    p0 = get_start_point(spnta, args.initial_sample_spread)
    # Diagonal starter covariance in internal coordinates.
    cov = np.diag(np.maximum(np.abs(p0), 1.0) ** 2 * 1e-4)

    chain_dir = f"{args.outdir}/ptmcmc_chains"
    if not os.path.isdir(chain_dir):
        os.mkdir(chain_dir)

    sampler = PTSampler(
        spnta.ndim,
        spnta.lnlike,
        spnta.lnprior,
        cov,
        outDir=chain_dir,
        resume=False,
    )
    cov_update = max(1, min(args.burnin, 1000))
    sampler.sample(
        p0,
        args.nsteps,
        burn=args.burnin,
        thin=args.thin,
        covUpdate=cov_update,
    )

    # PTMCMCSampler writes the cold chain to chain_1[.0].txt.
    chain_file = None
    for fname in ("chain_1.0.txt", "chain_1.txt"):
        fpath = f"{chain_dir}/{fname}"
        if os.path.isfile(fpath):
            chain_file = fpath
            break
    if chain_file is None:  # pragma: no cover
        raise FileNotFoundError(
            f"Unable to find PTMCMCSampler cold chain file in {chain_dir}"
        )

    chain = np.atleast_2d(np.loadtxt(chain_file))
    if chain.shape[1] < spnta.ndim:  # pragma: no cover
        raise ValueError(
            f"PTMCMCSampler chain has only {chain.shape[1]} columns, expected at least {spnta.ndim} parameter columns."
        )
    samples_raw = chain[:, : spnta.ndim]
    sampler_info = {
        "sampler": "ptmcmc",
        "nsteps": args.nsteps,
        "burnin": args.burnin,
        "thin": args.thin,
        "covUpdate": cov_update,
        "chain_dir": "ptmcmc_chains",
        "chain_file": os.path.basename(chain_file),
        "vectorized": False,
    }
    return samples_raw, sampler_info


def get_start_samples(spnta: SPNTA, s: float, nwalkers: int) -> np.ndarray:
    """Get starting samples for the MCMC. nwalkers is the number of samples
    to be returned."""

    p0_ = np.array(
        [spnta.prior_transform(cube) for cube in np.random.rand(nwalkers, spnta.ndim)]
    )
    p0 = (
        ((1 - s) * spnta.maxpost_params + s * p0_)
        if np.isfinite(spnta.lnpost(spnta.default_params))
        else p0_
    )
    return p0


def get_start_point(spnta: SPNTA, s: float) -> np.ndarray:
    """Get one PTMCMC start point in internal coordinates."""
    p0_ = np.array(spnta.prior_transform(np.random.rand(spnta.ndim)))
    return (
        (1 - s) * spnta.maxpost_params + s * p0_
        if np.isfinite(spnta.lnpost(spnta.default_params))
        else p0_
    )
