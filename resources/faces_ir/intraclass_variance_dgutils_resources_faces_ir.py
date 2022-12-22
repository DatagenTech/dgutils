import datagen.api as datapoint_api
from datagen.api import catalog, assets
from datagen.api.catalog import attributes
from datagen.api.catalog.attributes import Gender, Ethnicity, Age
import random
import numpy as np
random.seed(5263)


##### PARAMETERS DEFINITION #######
# Change that to your JSON path
DST_JSON_PATH = 'dgutils_resources_faces_ir.json'


###### RANDOMIZING FUNCTIONS: For each of the assets we pick, we randomize the controls ######
def random_hair_color():
    return assets.HairColor(melanin=random.random(), redness=random.random(), whiteness=random.random(),
                            roughness=random.uniform(0.15, 0.5), index_of_refraction=random.uniform(1.4, 1.65))


def random_expression():
    return assets.Expression(name=random.choice(list(assets.ExpressionName)), intensity=random.uniform(0.1, 1))


def randomize_glasses_controls(glasses):
    glasses.frame_color = random.choice(list(assets.FrameColor))
    glasses.frame_metalness = random.random()
    glasses.lens_color = random.choice(list(assets.LensColor))
    glasses.lens_reflectivity = random.random()
    glasses.lens_transparency = random.random()
    glasses.position = random.choice(list(assets.GlassesPosition))


def randomize_mask_controls(mask):
    mask.color = random.choice(list(assets.MaskColor))
    mask.texture = random.choice(list(assets.MaskTexture))
    mask.position = random.choice(list(assets.MaskPosition))
    mask.roughness = random.random()


def randomize_background_controls(background):
    background.rotation = random.uniform(0, 360)


def randomize_eyes(eyes):
    GAZE_MIN_Z = -0.5
    GAZE_MAX_Z = 0.3
    gaze_direction = np.array(
        [random.uniform(-0.5, 0.5), random.uniform(-1, -0.85), random.uniform(GAZE_MIN_Z, GAZE_MAX_Z)])
    eyes.target_of_gaze = assets.Gaze(distance=random.uniform(0.3, 6), direction=assets.Vector(x=gaze_direction[0],
                                                                                               y=gaze_direction[1],
                                                                                               z=gaze_direction[2]))
    if gaze_direction[2] > 0:
        min_closure = 0
        max_closure = 1 - (gaze_direction[2] / GAZE_MAX_Z)
    else:
        min_closure = -gaze_direction[2]
        max_closure = 1

    eyes.eyelid_closure = random.uniform(min_closure, max_closure)
    pass

datapoints = []

######  MAIN LOOP ######
# Picking up humans and backgrounds from catalog

# dgutils/resources/faces_ir/1
# You can adjust your camera parameters here
camera = assets.Camera(
    name='camera',
    intrinsic_params=assets.IntrinsicParams(
        projection=assets.Projection.PERSPECTIVE,
        resolution_width=1024,
        resolution_height=1024,
        fov_horizontal=25,
        fov_vertical=25,
        wavelength=assets.Wavelength.NIR
    ),
    extrinsic_params=assets.ExtrinsicParams(
        location=assets.Point(x=0.0, y=-0.6, z=0.12),
        rotation=assets.CameraRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
)

lights = [
    assets.Light(
        light_type='nir',
        beam_angle=25.0,
        falloff=0.0,
        brightness=10.0,
        location=assets.Point(x=0.0, y=-0.6, z=0.12),
        rotation=assets.LightRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
]

# Defining the identity class we are picking for this iteration
human = random.choices(catalog.humans.get(gender=Gender.MALE, age=Age.OLD, ethnicity=Ethnicity.MEDITERRANEAN))[0]

# Adding accessories
glasses = None
mask = None

human.head.rotation = assets.HeadRotation(yaw=-25.437811969667525, pitch=-7.5400338324980325, roll=0.0)

human.head.expression = assets.Expression(name=assets.ExpressionName.NEUTRAL, intensity=0.1)
human.head.eyes.eyelid_closure = 0.0

kwargs = {'human': human, 'camera': camera, 'lights': lights}
if mask is not None:
    kwargs['mask'] = mask
if glasses is not None:
    kwargs['glasses'] = glasses

datapoints.append(datapoint_api.create_datapoint(**kwargs))


# dgutils/resources/faces_ir/2
camera = assets.Camera(
    name='camera',
    intrinsic_params=assets.IntrinsicParams(
        projection=assets.Projection.PERSPECTIVE,
        resolution_width=1024,
        resolution_height=1024,
        fov_horizontal=25,
        fov_vertical=25,
        wavelength=assets.Wavelength.NIR
    ),
    extrinsic_params=assets.ExtrinsicParams(
        location=assets.Point(x=0.0, y=-0.6, z=0.12),
        rotation=assets.CameraRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
)

lights = [
    assets.Light(
        light_type='nir',
        beam_angle=25.0,
        falloff=0.0,
        brightness=10.0,
        location=assets.Point(x=0.0, y=-0.6, z=0.12),
        rotation=assets.LightRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
]

# Defining the identity class we are picking for this iteration
human = random.choices(catalog.humans.get(gender=Gender.MALE, age=Age.YOUNG, ethnicity=Ethnicity.NORTH_EUROPEAN))[0]

# Adding accessories
glasses = random.choice(catalog.glasses.get(gender=human.attributes.gender))
glasses.id = "484f76b7-9ca0-476f-9c03-578b3ba0ecb6"
glasses.lens_color = assets.LensColor.YELLOW
glasses.lens_reflectivity = 0.0
glasses.lens_transparency = 1.0
glasses.position = assets.GlassesPosition.ON_NOSE

mask = None

human.head.rotation = assets.HeadRotation(yaw=-0.20807154232265823, pitch=-3.1993882396144615, roll=0.0)

human.head.expression = assets.Expression(name=assets.ExpressionName.NEUTRAL, intensity=0.1)
human.head.eyes.eyelid_closure = 0.0

kwargs = {'human': human, 'camera': camera, 'lights': lights}
if mask is not None:
    kwargs['mask'] = mask
if glasses is not None:
    kwargs['glasses'] = glasses

datapoints.append(datapoint_api.create_datapoint(**kwargs))


# dgutils/resources/faces_ir/2 - vis
camera = assets.Camera(
    name='camera',
    intrinsic_params=assets.IntrinsicParams(
        projection=assets.Projection.PERSPECTIVE,
        resolution_width=1024,
        resolution_height=1024,
        fov_horizontal=25,
        fov_vertical=25,
        wavelength=assets.Wavelength.VISIBLE
    ),
    extrinsic_params=assets.ExtrinsicParams(
        location=assets.Point(x=0.0, y=-0.6, z=0.12),
        rotation=assets.CameraRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
)

background = random.choice(catalog.backgrounds.get())
randomize_background_controls(background)

kwargs = {'human': human, 'camera': camera, 'background': background}
if mask is not None:
    kwargs['mask'] = mask
if glasses is not None:
    kwargs['glasses'] = glasses

datapoints.append(datapoint_api.create_datapoint(**kwargs))


datapoint_api.dump(assets.DataRequest(datapoints=datapoints), path=DST_JSON_PATH)
