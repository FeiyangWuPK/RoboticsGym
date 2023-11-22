import omni.kit.commands
from pxr import UsdLux, Sdf, Gf, UsdPhysics, PhysicsSchemaTools


def convert_cassie_xml_to_usd(cassie_file_path: str):
    # create new stage
    omni.usd.get_context().new_stage()

    # setting up import configuration:
    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    import_config.set_fix_base(False)
    import_config.set_make_default_prim(False)

    # import MJCF
    omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=cassie_file_path,
        import_config=import_config,
        prim_path="/ant",
    )

    # get stage handle
    stage = omni.usd.get_context().get_stage()

    # enable physics
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))

    # set gravity
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)

    # add lighting
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(500)


def convert_digit_xml_to_usd(digit_file_path: str):
    # create new stage
    omni.usd.get_context().new_stage()

    # setting up import configuration:
    status, import_config = omni.kit.commands.execute("MJCFCreateImportConfig")
    import_config.set_fix_base(False)
    import_config.set_make_default_prim(False)

    # import MJCF
    omni.kit.commands.execute(
        "MJCFCreateAsset",
        mjcf_path=digit_file_path,
        import_config=import_config,
        prim_path="/ant",
    )

    # get stage handle
    stage = omni.usd.get_context().get_stage()

    # enable physics
    scene = UsdPhysics.Scene.Define(stage, Sdf.Path("/physicsScene"))

    # set gravity
    scene.CreateGravityDirectionAttr().Set(Gf.Vec3f(0.0, 0.0, -1.0))
    scene.CreateGravityMagnitudeAttr().Set(981.0)

    # add lighting
    distantLight = UsdLux.DistantLight.Define(stage, Sdf.Path("/DistantLight"))
    distantLight.CreateIntensityAttr(500)


if __name__ == "__main__":
    cassie_file_path = "/home/feiyang/Documents/Repositories/RoboticsGym/roboticsgym/envs/xml/mj_cassie.xml"
    convert_cassie_xml_to_usd(cassie_file_path)

    digit_file_path = "/home/feiyang/Documents/Repositories/RoboticsGym/roboticsgym/envs/xml/digit_v3.xml"
    convert_digit_xml_to_usd(digit_file_path)
