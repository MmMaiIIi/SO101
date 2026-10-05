from deployment.pi05_stack2.config import DeploymentConfig


def test_hardware_disabled_by_default():
    config = DeploymentConfig()
    assert not config.hardware_enabled
    config.validate()
