FROM ubuntu:latest

RUN apt update && apt install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    libssl-dev \
    libffi-dev \
    python3-setuptools \
    python3-venv \
    python3-wheel \
    git \
    vim

RUN python3 -m pip install --upgrade pip

RUN python3 -m pip install \
    absl-py \
    asttokens \
    astunparse \
    backcall \
    cachetools \
    certifi \
    charset-normalizer

RUN python3 -m pip install \
    contourpy \
    cycler \
    decorator \
    executing \
    flatbuffers \
    fonttools \
    gast \
    google-auth \
    google-auth-oauthlib \
    google-pasta

RUN python3 -m pip install \
    grpcio \
    h5py \
    idna \
    ipython \
    jedi \
    keras \
    kiwisolver \
    libclang \
    Markdown \
    MarkupSafe \
    matplotlib \
    matplotlib-inline \
    numpy \
    oauthlib

RUN python3 -m pip install \
    opt-einsum \
    packaging \
    parso \
    pexpect \
    pickleshare \
    Pillow \
    prompt-toolkit \
    protobuf \
    ptyprocess \
    pure-eval \
    pyasn1 \
    pyasn1-modules \
    Pygments \
    pyparsing \
    python-dateutil \
    requests \
    requests-oauthlib \
    rsa \
    scipy \
    six \
    stack-data \
    seaborn \
    joblib \
    numba \
    audioread

RUN python3 -m pip install \
    tensorboard \
    tensorboard-data-server \
    tensorboard-plugin-wit \
    tensorflow \
    tensorflow-estimator \
    tensorflow-hub \
    tensorflow-io-gcs-filesystem \
    termcolor \
    traitlets \
    typing_extensions \
    urllib3 \
    wcwidth \
    Werkzeug \
    wrapt

RUN python3 -m pip install \
    scikit-learn \
    pandas

RUN python3 -m pip install \
    librosa

RUN apt install -y libsndfile1-dev
