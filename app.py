import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2

import tensorflow as tf
import streamlit as st

#import keras
from tensorflow.keras.models import load_model
import pickle

from PIL import Image

#Loading the Model

model = load_model('Capstone-Model-VGG19.h5')

#Name of Classes
a_file = open("plant_diseases.pkl", "rb")
ref = pickle.load(a_file)

a_file.close()


#Setting Title of App
#st.title("Vehicle Classification")


html_temp = """
<div style="background-color:dodgerblue;padding:10px">
<h2 style="color:white;text-align:center;">Plant Disease Classification App </h2>
</div>
    """
st.markdown(html_temp,unsafe_allow_html=True)

st.markdown("")
image = Image.open('Tomato_img1.jpg')
st.image(image,use_column_width=True)

st.markdown("""
This web page leverages deep learning to classify plant disease images for the following crops:

* Maize / Corn
* Tomatoes

Upload an image of a diseased plant below to identify the plant disease.
"""
)


#Uploading the image
plant_image = st.file_uploader("Upload Image below...", type=["jpg", "jpeg", "png"])
submit = st.button('Identify Disease')

with st.spinner('Identfying...'):
  #On predict button click
  if submit:


      if plant_image is not None:
        # Convert the file to an opencv image.
        file_bytes = np.asarray(bytearray(plant_image.read()), dtype=np.uint8)
        opencv_image = cv2.imdecode(file_bytes, 1)



        # Displaying the image
        st.image(opencv_image, channels="BGR",width=512)
        
        #Resizing the image
        opencv_image = cv2.resize(opencv_image, (256,256))
    
        #Convert image to 4 Dimension
        opencv_image.shape = (1,256,256,3)

        #Make Prediction
        pred = np.argmax(model.predict(opencv_image))
        prediction = ref[pred]
        #st.subheader(str("Disease Identified as "+prediction))
        
        ## Additional info
        if prediction == "Maize_Blight":
            st.subheader("Disease Name")
            st.markdown('Maize Blight')
            st.subheader("Disease Cause")
            st.markdown('Fungus (Exserohilum Turcicum). Fungal spores are carried by insects, wind, water and animals from infected farms.')
            st.subheader("Disease Management")
            st.markdown("""
            - Planting hybrids that are resistant.
            - Spraying fungicides.
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Long narrow, tan lesions that form parallel to leaf margins.
            - Small light green to grayish spots approximately 1-2 weeks after infection. 
            """)
            
        elif prediction == "Maize_Common_Rust":
            st.subheader("Disease Name")
            st.markdown('Maize Common Rust')
            st.subheader("Disease Cause")
            st.markdown('Fungus (Puccinia sorghi).')
            st.subheader("Disease Management")
            st.markdown("""
            - Planting hybrids resistant varieties
            - Spraying fungicides
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - circular pustules, powdery, brown becoming brown-black as the plant matures.
            - The pustules occur on all above-ground parts, but are most common on the leaves where they are scattered on both surfaces.
            - In severe cases, the leaves and leaf sheaths turn yellow and die early
            """)
            
        elif prediction == "Maize_Gray_Leaf_Spot" :
            st.subheader("Disease Name")
            st.markdown('Maize Gray Leaf Spot')
            st.subheader("Disease Cause")
            st.markdown('Fungal Pathogens (Cercospora zeae-maydis).')
            st.subheader("Disease Management")
            st.markdown("""
            - Crop roatation.
            - Weed control.
            - Spraying fungicides
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Symptoms are initially first observed on the lower leaves of the maize plant.
            - Lesions first appear as small tan spots about 1 to 3 mm in size and are irregular in shape.
            -  tan spots usually have yellow or chlorotic borders and, are more easily observed when the leaf is held to light.
            - As disease progresses, lesions coalesce and blighting of the whole leaf may result.
            """)
            
        elif prediction == "Maize_Healthy":
            st.subheader("Disease Name")
            st.markdown('Good News!!! Your plant is healthy. Keep up the great work.')
              
        elif prediction == "Tomato_Bacterial_spot":
            st.subheader("Disease Name")
            st.markdown('Tomato Bacterial Spot')
            st.subheader("Disease Cause")
            st.markdown('Bacterial Pathogens (Xanthomonas vesicatoria).')
            st.subheader("Disease Management")
            st.markdown("""
            - Do not use seed from infected plants.
            - Care in handling seedlings during transplanting.
            - Spraying copper fungicides
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Look for many small (2-3 mm) irregular spots on the leaves, leaf stalks, stems and fruits.
            - Spots on the leaf stalks and stems are elongate.
            - Look for scabby spots on the fruits with transparent margins.
            - Leaves looked scorched and turn yellow.
            """)
            
        elif prediction == "Tomato_Early_blight":
            st.subheader("Disease Name")
            st.markdown('Tomato Early Blight')
            st.subheader("Disease Cause")
            st.markdown('Fungus (Alternaria tomatophila). Spores can be spread throughout a field by wind, human contact or equipment, resulting in many reinfection opportunities throughout a growing season.')
            st.subheader("Disease Management")
            st.markdown("""
            - Plant resistant varieties.
            - Crop rotation.
            - Mulching.
            - Prunning bottom leaves to improve airflow.
            - Spray protectant fungicides, chlorothalonil or copper products.
            
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - small dark spots form on older foliage near the ground. Leaf spots are round, brown and can grow up to 1/2 inch in diameter.
            - Larger spots have target-like concentric rings. The tissue around spots often turns yellow.
            - Severely infected leaves turn brown and fall off, or dead, dried leaves may cling to the stem.
            - Fruit spots are leathery and black, with raised concentric ridges. They generally occur near the stem. Infected fruit may drop from the plant.
            """)
            
        elif prediction == "Tomato_Late_blight" :
            st.subheader("Disease Name")
            st.markdown('Tomato Late Blight')
            st.subheader("Disease Cause")
            st.markdown('Fungus like micro-organism (Phytophthora infestans). ')
            st.subheader("Disease Management")
            st.markdown("""
            - Avoid overhead irrigation.
            - Prune few branches from lower parts of plant to improve airflow.
            - Prune infected leaves.
            - Stake plants.
            - Preventive sprays e.g copper products and fungicides.
            
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Look for spots and patches on the leaves which grow rapidly and produce a furry white growth on the underside. 
            - Black or brown irregular shaped spots.
            - Leaves turn yellow shrivel and die.
            - Dark brown, firm rots occur on the fruits of tomatoes.
            """)
            
        elif prediction == "Tomato_Leaf_Mold":
            st.subheader("Disease Name")
            st.markdown('Tomato Leaf Mold')
            st.subheader("Disease Cause")
            st.markdown('Fungus (Passalora fulva). The disease is driven by high relative humidity (greater than 85%).')
            st.subheader("Disease Management")
            st.markdown("""
            - Use drip irrigation to avoid watering folliage.
            - Stake and prune branches to improve airflow.
            - Remove crop residue at the end of season to avoid carrying over pests and diseases.
            - Spraying fungicides.
            
            
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - The oldest leaves are infected first. 
            - Pale greenish-yellow spots, usually less than 1/4 inch, with no definite margins, form on the upper sides of leaves.
            - Olive-green to brown velvety mold forms on the lower leaf surface below leaf spots.
            - Leaf spots grow together and turn brown. Leaves wither and die but often remain attached to the plant.
            - Infected blossoms turn black and fall off.
            - Fruit infections start as a smooth black irregular area on the stem end of the fruit. As the disease progresses, the infected area becomes sunken, dry and leathery.
            """)
            
        elif prediction == "Tomato_Septoria_leaf_spot":
            st.subheader("Disease Name")
            st.markdown('Tomato Septoria Leaf Spot')
            st.subheader("Disease Cause")
            st.markdown('Fungus (Septoria lycopersici).')
            st.subheader("Disease Management")
            st.markdown("""
            - Prune diseased leaves.
            - Prune branches to improve airflow.
            - Mulch around base.
            - Avoid overhead irrigation.
            - Control weeds.
            - Crop rotation.
            - Fungicidal sprays.
            
            
            """)
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Septoria leaf spot usually appears on the lower leaves after the first fruit sets. 
            - Spots are circular, about 1/16 to 1/4 inch in diameter with dark brown margins and tan to gray centers with small black fruiting structures.
            - Characteristically, there are many spots per leaf.
            - Disease spreads upwards from oldest to youngest growth.
            - If leaf lesions are numerous, the leaves turn slightly yellow, then brown, and then wither.
            - Fruit infection is rare.
            """)
            
        elif prediction == "Tomato_Spider_mites Two-spotted_spider_mite":
            st.subheader("Disease Name")
            st.markdown('Tomato Spider Mites Two Sotted Spider Mite')
            st.subheader("Disease Cause")
            st.markdown('Spider Mites (Tetranychus urticae). Outbreaks are often caused by the use of broad-spectrum insecticides which interfere with the numerous natural enemies that help to manage mite populations.
            st.subheader("Disease Management")
            st.markdown("""
            - Overhead irrigation helps reduce mite population.
            - Weed in and around nurseries, especially remove wild plants.
            - Avoid water-stressed plants as they encourage mite outbreaks.
            - Check plants in the nursery for mite infestations, and if pressent do not plant in the field.
            - Do not plant downwind from infested crops.
            - If pesticides are used, they should be applied carefully. Rotate between different chemical groups, to prevent resistance developing to any one of them.
            - Use pesticidal oils (white oil based on vegetable oils), horticultural oil (based on petroleum oils) or soaps.
            """)
                        
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Spider mites have needle-like mouthparts and use them to suck juice from the leaves. 
            - Two-spotted mites make webs (like spiders) on the under surface of leaves.
            - Adult females are approximately 1/50-inch long, slightly orange or pale green in color with two dark spots on their body.
            - Withering or spotting of leaves.
            - Yellowing of the plant or some of the leaves.
            - Loss of folliage and death.
            """)
                        
        elif prediction == "Tomato_Target_Spot":
            st.subheader("Disease Name")
            st.markdown('Tomato Target Spot')
            st.subheader("Disease Cause")
            st.markdown('Fungus (Corynespora cassiicola). 
            st.subheader("Disease Management")
            st.markdown("""
            - Remove old plant debris.
            - Plant as far as possible from papaya.
            - Check all seedlings in the nursery, and throw away any with leaf spots.
            - Remove a few branches from the lower part of the plants to allow better airflow at the base.
            - Remove and burn the lower leaves as soon as the disease is seen, especially after the lower fruit trusses have been picked.
            - Do not use overhead irrigation; otherwise, it will create conditions for spore production and infection.
            - Spraying fungicides such as copper oxychloride.
            """)
                        
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Disease starts on the older leaves and spreads upwards. 
            - Irregular-shaped spots (less than 1 mm) with a yellow margin.
            - Some of the spots enlarge up to 10 mm and show characteristics rings, hence the name of 'target spot'.
            - Spread to all leaflets and to other leaves is rapid causing the leaves to turn yellow, collapse and die 

            """)
                        
        elif prediction == "Tomato_Tomato_Yellow_Leaf_Curl_Virus":
            st.subheader("Disease Name")
            st.markdown('Tomato Yellow Leaf Curl Virus')
            st.subheader("Disease Cause")
            st.markdown('Silverleaf white fly (Bemisia tabaci). 
            st.subheader("Disease Management")
            st.markdown("""
            - Crop rotation
            - A trap can be made with a piece of board 12 inches x 12 inches painted bright yellow. Spread petroleum jelly or Biotac on it. The yellow colour attracts the whiteflies to the boards and they stick to them. The boards are placed at the height of the plants. Monitor all crops, not just tomato plants as the whitefly may have passed the virus onto another crop.
            - Plants that show signs of the virus after 3-4 weeks of transplanting should be bagged (to prevent the whiteflies leaving), uprooted and burned to reduce spread of the virus.
            - Plants should be watered and fertilized adequately to reduce stress and to build plant health.
            - Imidacloprid should be sprayed on the entire plant and below the leaves; eggs and flies are often found below the leaves. Spray every 14-21 days and rotate on a monthly basis with Abamectin so that the whiteflies do not build-up resistance to chemicals.
            - Chemicals are most effective when used early in the morning and late in the evening. This is because insects are most active at these times of the day.

            """)
                        
            st.subheader("Disease Symptoms")
            st.markdown("""
            - Leaf yellowing. 
            - Leaf curling upward.
            - Reduced leaflet size.
            - Flowers dropping.

            """)
                        
        else: 
            #Body of Tomato healthy
            st.subheader("Disease Name")
            st.markdown('Good News!!! Your plant is healthy. Keep up the great work.')
            
