from datagen.api import catalog, datapoint as datapoint_api
from datagen.api.datapoint import assets
from datagen.api.catalog import attributes
from datagen.api.catalog.attributes import Gender, Ethnicity, Age
import random
import numpy as np
random.seed(5263)


##### PARAMETERS DEFINITION #######
# Number of identities and number of variations per ID
NROF_IDS = 100
NROF_VARIATIONS = 1
# Probability to have an accessory (glasses or mask)
ACCESSORY_PROB = 0.0
# Has effect only if we have accessories
GLASSES_PROB = 0.0
# Has effect only if male without mask
BEARD_PROB = 0.0
# Change that to your JSON path
DST_JSON_PATH = 'dgutils_resources_faces_nn.json'

###### RANDOMIZING FUNCTIONS: For each of the assets we pick, we randomize the controls ######
def random_hair_color():
    return assets.HairColor(melanin=random.random(), redness=random.random(), whiteness = random.random(),
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


######  MAIN LOOP ######
# Picking up humans and backgrounds from catalog
backgrounds = catalog.backgrounds.get()

# You can adjust your camera parameters here
camera = assets.Camera(
    name='camera',
    intrinsic_params=assets.IntrinsicParams(
        projection=assets.Projection.PERSPECTIVE,
        resolution_width=128,
        resolution_height=128,
        fov_horizontal=8,
        fov_vertical=8,
        wavelength=assets.Wavelength.VISIBLE
    ),
    extrinsic_params=assets.ExtrinsicParams(
        location=assets.Point(x=0.0, y=-1.6, z=0.12),
        rotation=assets.CameraRotation(yaw=0.0, pitch=0.0, roll=0.0)
    )
)

datapoints = []

# We randomize: Backgrounds, head pose, expression, glasses, masks, eye openness, beards, hair whiteness
for i in range(NROF_IDS):
    print(f'Processing datapoint number {str(i)}...')
    # Defining the identity class we are picking for this iteration
    # Here the ID class is sampled uniformly. You can change the weights to get a different distribution
    gender = random.choices([Gender.MALE, Gender.FEMALE], weights=[1 / 2, 1 / 2])[0]
    age = random.choices([Age.YOUNG, Age.ADULT, Age.OLD], weights=[1 / 3, 1 / 3, 1 / 3])[0]
    ethnicity = random.choices([Ethnicity.AFRICAN, Ethnicity.NORTH_EUROPEAN, Ethnicity.MEDITERRANEAN,
                                Ethnicity.HISPANIC, Ethnicity.SOUTH_ASIAN, Ethnicity.SOUTHEAST_ASIAN],
                               weights=[1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6, 1 / 6])[0]

    human = random.choices(catalog.humans.get(gender=gender, age=age, ethnicity=ethnicity))[0]


    hair_color = random_hair_color()
    human.head.hair = random.choice(catalog.hair.get(age_group_match=human.attributes.age,
                                                     gender_match=human.attributes.gender,
                                                     ethnicity_match=human.attributes.ethnicity))
    human.head.eyebrows = random.choice(catalog.eyebrows.get(gender_match=human.attributes.gender))
    human.head.hair.color_settings = hair_color
    human.head.eyebrows.color_settings = hair_color
    human.head.eyes = random.choice(catalog.eyes.get())
    for _ in range(NROF_VARIATIONS):
        # We play with the hair whiteness
        hair_color.whiteness = random.random()

        # Adding accessories
        glasses = None
        mask = None
        if random.random() < ACCESSORY_PROB:
            if random.random() < GLASSES_PROB:
                glasses = random.choice(catalog.glasses.get(gender=human.attributes.gender))
                randomize_glasses_controls(glasses)
            else:
                mask = random.choice(catalog.masks.get(gender=human.attributes.gender))
                randomize_mask_controls(mask)

        if mask is None and human.attributes.gender == attributes.Gender.MALE and random.random() < BEARD_PROB:
            human.head.facial_hair = random.choice(catalog.beards.get())
            human.head.facial_hair.color_settings = hair_color
        else:
            human.head.facial_hair = None

        human.head.rotation = assets.HeadRotation(yaw=0.0, pitch=0.0, roll=0.0)
        background = random.choice(backgrounds)
        randomize_background_controls(background)

        human.head.expression = random_expression()
        randomize_eyes(human.head.eyes)

        kwargs = {'human': human, 'camera': camera, 'background': background}
        if mask is not None:
            kwargs['mask'] = mask
        if glasses is not None:
            kwargs['glasses'] = glasses
        datapoints.append(datapoint_api.create_datapoint(**kwargs))


datapoint_api.dump(assets.DataRequest(datapoints=datapoints), path=DST_JSON_PATH)

