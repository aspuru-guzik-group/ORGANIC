FROM  beangoben/pimp_jupyter

USER root
ADD requirements.yml /home/jovyan
RUN conda env update -n root -f requirements.yml && \
    conda clean --all