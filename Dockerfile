FROM tensorflow/tensorflow:1.15.0-gpu-py3
# Configure environment variables
ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
#COPY TrainingSet_2mm/ /app/
#COPY ValidationSet_2mm/ ../ValidationSet_2mm
#COPY TrainingSet_NoISO/ ../TrainingSet_NoISO
#COPY Training_SET_Non_ISO_2/ ../Training_SET_Non_ISO_2
#COPY TrainingSet_ISO_NS/ ../TrainingSet_ISO_NS

#COPY ValidationSet_NoISO/ ../ValidationSet_NoISO
#COPY Validation_SET_Non_ISO_2/ ../Validation_SET_Non_ISO_2
#COPY ValidationSet_ISO_NS/ ../ValidationSet_ISO_NS

#COPY TrainingSet_Masks_NoISO/ ../TrainingSet_Masks_NoISO
#COPY Training_SET_MASKS_Non_ISO_2/ ../Training_SET_MASKS_Non_ISO_2
#COPY TrainingSet_MASKS_ISO_NS/ ../TrainingSet_MASKS_ISO_NS

#COPY ValidationSet_Masks_NoISO/ ../ValidationSet_Masks_NoISO
#COPY Validation_SET_MASKS_Non_ISO_2/ ../Validation_SET_MASKS_Non_ISO_2
#COPY ValidationSet_MASKS_ISO_NS/ ../ValidationSet_MASKS_ISO_NS

COPY setup.py /app/setup.py
WORKDIR /app
RUN pip3 install -e .
COPY src/ /app/
# Copy scripts
#WORKDIR /app/src
#COPY --chown=user src/ /app/
#COPY src/ /app/
#COPY ./ /app/

# Define entry point
#USER user

#RUN apt-get update && apt-get install -y python3-pip
#RUN apt-get install -y python3-pip

#ENTRYPOINT ["python","-u", "HelloWorld.py"]
ENTRYPOINT ["python","-u", "task_laucher.py"]
