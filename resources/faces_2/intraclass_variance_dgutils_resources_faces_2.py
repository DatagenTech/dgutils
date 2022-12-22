import datagen.api as datapoint_api
from datagen.api import catalog, assets
from datagen.api.catalog import attributes
from datagen.api.catalog.attributes import Gender, Ethnicity, Age
import random
import numpy as np
random.seed(5263)


##### PARAMETERS DEFINITION #######
DST_JSON_PATH = 'dgutils_resources_faces_2.json'


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
backgrounds = catalog.backgrounds.get()

# dgutils/resources/faces_2/1
# You can adjust your camera parameters here
camera = assets.Camera(
    name='camera',
    intrinsic_params=assets.IntrinsicParams(
        projection=assets.Projection.PERSPECTIVE,
        resolution_width=1024,
        resolution_height=1024,
        fov_horizontal=10,
        fov_vertical=10,
        wavelength=assets.Wavelength.VISIBLE
    ),
    extrinsic_params=assets.ExtrinsicParams(
        location=assets.Point(x=0.0, y=-1.6, z=0.12),
        rotation=assets.CameraRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
)

# Defining the identity class we are picking for this iteration
human = random.choices(catalog.humans.get(gender=Gender.FEMALE, age=Age.ADULT, ethnicity=Ethnicity.AFRICAN))[0]

# Adding accessories
glasses = None
mask = None

human.head.rotation = assets.HeadRotation(yaw=-10.971257731710612, pitch=15.769566368934827, roll=0.0)
background = random.choice(backgrounds)
randomize_background_controls(background)

human.head.expression = assets.Expression(name=assets.ExpressionName.NEUTRAL, intensity=0.1)
human.head.eyes.target_of_gaze = assets.Gaze(distance=6.0, direction=assets.Vector(x=0.35355, y=-0.86603, z=-0.35355))
human.head.eyes.eyelid_closure = 0.6

kwargs = {'human': human, 'camera': camera, 'background': background}
if mask is not None:
    kwargs['mask'] = mask
if glasses is not None:
    kwargs['glasses'] = glasses

datapoints.append(datapoint_api.create_datapoint(**kwargs))


# dgutils/resources/faces_2/2
camera = assets.Camera(
    name='camera',
    intrinsic_params=assets.IntrinsicParams(
        projection=assets.Projection.PERSPECTIVE,
        resolution_width=1024,
        resolution_height=1024,
        fov_horizontal=10,
        fov_vertical=10,
        wavelength=assets.Wavelength.VISIBLE
    ),
    extrinsic_params=assets.ExtrinsicParams(
        location=assets.Point(x=0.0, y=-1.6, z=0.12),
        rotation=assets.CameraRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
)

# Defining the identity class we are picking for this iteration
human = random.choices(catalog.humans.get(gender=Gender.MALE, age=Age.ADULT, ethnicity=Ethnicity.SOUTHEAST_ASIAN))[0]

# Adding accessories
glasses = None
mask = None

human.head.rotation = assets.HeadRotation(yaw=16.61665325998898, pitch=-11.628474755348517, roll=0.0)
background = random.choice(backgrounds)
randomize_background_controls(background)

human.head.expression = assets.Expression(name=assets.ExpressionName.HAPPINESS, intensity=0.2)
human.head.eyes.eyelid_closure = 0.0

kwargs = {'human': human, 'camera': camera, 'background': background}
if mask is not None:
    kwargs['mask'] = mask
if glasses is not None:
    kwargs['glasses'] = glasses

datapoints.append(datapoint_api.create_datapoint(**kwargs))


# dgutils/resources/faces_2/3
cameras = [
    assets.Camera(
        name='camera_1',
        intrinsic_params=assets.IntrinsicParams(
            projection=assets.Projection.PERSPECTIVE,
            resolution_width=1024,
            resolution_height=1024,
            fov_horizontal=10,
            fov_vertical=10,
            wavelength=assets.Wavelength.VISIBLE
        ),
        extrinsic_params=assets.ExtrinsicParams(
            location=assets.Point(x=0.0, y=-1.3499999046325684, z=1.5099999904632568),
            rotation=assets.CameraRotation(yaw=0.0, pitch=-45.3, roll=0.0)
        )
    ),
    assets.Camera(
        name='camera_2',
        intrinsic_params=assets.IntrinsicParams(
            projection=assets.Projection.PERSPECTIVE,
            resolution_width=1024,
            resolution_height=1024,
            fov_horizontal=10,
            fov_vertical=10,
            wavelength=assets.Wavelength.VISIBLE
        ),
        extrinsic_params=assets.ExtrinsicParams(
            location=assets.Point(x=0.0, y=1.7599999904632568, z=0.9399999976158142),
            rotation=assets.CameraRotation(yaw=180.0, pitch=-24.7, roll=0.0)
        )
    ),
    assets.Camera(
        name='camera_3',
        intrinsic_params=assets.IntrinsicParams(
            projection=assets.Projection.PERSPECTIVE,
            resolution_width=1024,
            resolution_height=1024,
            fov_horizontal=10,
            fov_vertical=10,
            wavelength=assets.Wavelength.VISIBLE
        ),
        extrinsic_params=assets.ExtrinsicParams(
            location=assets.Point(x=-2.190000057220459, y=-0.5700000524520874, z=0.12999999523162842),
            rotation=assets.CameraRotation(yaw=75.0, pitch=0.0, roll=0.0)
        )
    ),
]

# Defining the identity class we are picking for this iteration
human = random.choices(catalog.humans.get(gender=Gender.MALE, age=Age.ADULT, ethnicity=Ethnicity.SOUTHEAST_ASIAN))[0]

# Adding accessories
glasses = None
mask = None

human.head.rotation = assets.HeadRotation(yaw=0.8966683014563318, pitch=-4.237275191252075, roll=0.0)
background = random.choice(backgrounds)
randomize_background_controls(background)

human.head.expression = assets.Expression(name=assets.ExpressionName.NEUTRAL, intensity=0.1)
human.head.eyes.eyelid_closure = 0.0

for cam in cameras:
    kwargs = {'human': human, 'camera': cam, 'background': background}
    if mask is not None:
        kwargs['mask'] = mask
    if glasses is not None:
        kwargs['glasses'] = glasses

    datapoints.append(datapoint_api.create_datapoint(**kwargs))


datapoint_api.dump(assets.DataRequest(datapoints=datapoints), path=DST_JSON_PATH)
