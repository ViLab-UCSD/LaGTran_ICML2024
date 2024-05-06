#%%
import openai
import json
import pathlib
import numpy as np
import time
openai.api_key = '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'

# places205 = json.load(open("/home/tarun/metadata/geoPlaces_metadata.json"))['categories']
# classnames = [p["category_name"] for p in places205]
# cnames = [c.replace("_indoor","").replace("_outdoor","").replace("_"," ") for c in classnames]

places205 = json.load(open("/home/tarun/metadata/geoImnet_metadata.json"))['categories']
classnames = [p["category_name"] for p in places205]
cnames = [c.replace("+"," ").replace("_"," ") for c in classnames]

#%% Generate Prompts.
def generate_prompt(category_name: str):
    # you can replace the examples with whatever you want; these were random and worked, could be improved
    return f"""Q: What are useful visual features for distinguishing a lemur in a photo?
A: There are several useful visual features to tell there is a lemur in a photo:
- four-limbed primate
- black, grey, white, brown, or red-brown
- wet and hairless nose with curved nostrils
- long tail
- large eyes
- furry bodies
- clawed hands and feet
Q: What are useful visual features for distinguishing a television in a photo?
A: There are several useful visual features to tell there is a television in a photo:
- electronic device
- black or grey
- a large, rectangular screen
- a stand or mount to support the screen
- one or more speakers
- a power cord
- input ports for connecting to other devices
- a remote control
Q: What are useful features for distinguishing a {category_name} in a photo?
A: There are several useful visual features to tell there is a {category_name} in a photo:
-
"""

prompts = [generate_prompt(_c) for _c in cnames]

#%% Query GPT-3.
def stringtolist(description):
    return [descriptor[2:] for descriptor in description.split('\n') if (descriptor != '') and (descriptor.startswith('- '))]

descriptions = []
for i in np.arange(0, len(prompts), 20):
    st = time.time()
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompts[i:i + 20],
        temperature=0.7,
        max_tokens=100,
    )
    print(i, ":", time.time() - st)
    descriptions += [stringtolist(_r["text"]) for _r in response["choices"]]

#%% Write generated descriptions to JSON.
descriptions_dict = {_c: _d for _c, _d in zip(classnames, descriptions)}
with open('descriptors_geoimnet.json', 'w') as outfile:
    outfile.write(json.dumps(descriptions_dict, indent=4))