from pathlib import Path

import pytest

from gwlens_mm.policy import FieldRole, InputPolicy, InputPolicyError

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture()
def policy():
    return InputPolicy.from_files(
        ROOT / "configs/policy/deployable_input_allowlist.json",
        ROOT / "configs/policy/privileged_input_denylist.json",
    )


def test_allowlisted_observations_pass(policy):
    assert policy.validate_model_inputs(
        [
            "gw_strain_primary",
            "observed_image_astrometry",
            "observed_time_difference",
            "observed_external_convergence_mean",
            "observed_external_convergence_std",
            "environment_modality_available",
        ]
    )


@pytest.mark.parametrize(
    "field",
    [
        "source_position_y_true",
        "source_plane_beta_x_true",
        "mu_abs_true",
        "clean_strain",
        "optimal_snr",
        "pair_id",
        "importance_weight",
        "A21",
        "trueMu",
        "oracleLensParameter",
        "cleanWaveform",
        "true_time_delay_seconds",
        "external_convergence_true",
        "kappa_ext_true",
        "stellar_anisotropy_true",
    ],
)
def test_forbidden_exact_and_alias_fields_fail_closed(policy, field):
    with pytest.raises(InputPolicyError, match="forbidden"):
        policy.validate_model_inputs([field])


def test_unknown_field_is_not_silently_accepted(policy):
    with pytest.raises(InputPolicyError, match="unknown"):
        policy.validate_model_inputs(["some_new_feature"])


def test_target_permission_does_not_imply_input_permission(policy):
    roles = {
        "mu_abs_true": FieldRole.TRAINING_TARGET,
        "observed_lens_redshift": FieldRole.MODEL_INPUT,
        "pair_id": FieldRole.GROUPING_PROVENANCE,
    }
    policy.validate_roles(roles)
    with pytest.raises(InputPolicyError):
        policy.validate_roles({"mu_abs_true": FieldRole.MODEL_INPUT})
