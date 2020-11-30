#FROM tensorflow/tensorflow
FROM ubuntu:latest

MAINTAINER Saeed

COPY ./requirements.txt /req/requirements.txt



RUN apt-get -y update && apt-get install -y --no-install-recommends \
         python3 \
		 python3-pip\
         nginx

#RUN set -xe \
#   && apt-get update \
#   && apt-get install -y python3-pip
# RUN pip3 install --upgrade pip



#RUN pip install --upgrade pip
RUN pip3 install -r /req/requirements.txt
EXPOSE 5000
COPY . /app
WORKDIR /app

ENTRYPOINT [ "python3" ]
CMD ["app.py"]


###################################################################################

