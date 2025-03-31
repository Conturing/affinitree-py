FROM quay.io/pypa/manylinux2014_x86_64:latest
ENV HOME="/root"
WORKDIR $HOME

# install python
RUN yum update -y \
    && yum install -y python3

# install stable version of rust
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y --profile minimal --default-toolchain none 
ENV PATH="$HOME/.cargo/bin:$PATH"
RUN rustup default stable

# configure virtual environment with build dependencies
RUN python3.10 -m venv $HOME/.venv \
    && .venv/bin/pip install \
    --upgrade pip \
    numpy \
    "maturin==1.5.1" \
    ziglang

# update path with local venv and all supported python interpreters
ENV PATH=$HOME/.venv/bin:/opt/python/cp38-cp38/bin:/opt/python/cp39-cp39/bin:/opt/python/cp310-cp310/bin:/opt/python/cp311-cp311/bin:/opt/python/cp312-cp312/bin:$PATH

# build crate cache (see https://github.com/PyO3/maturin/blob/main/Dockerfile)
ADD Cargo.toml ./Cargo.toml
ADD Cargo.lock ./Cargo.lock
RUN mkdir ./src \
    && touch ./src/lib.rs \
    && echo 'fn main() { println!("Dummy") }' > ./src/main.rs \
    && cargo build --release \
    && rm ./src/main.rs \
    && rm ./src/lib.rs

# copy relevant files for packaging
COPY python ./python
COPY src ./src
COPY pyproject.toml ./
COPY README.md ./
COPY LICENSE-APACHE ./
COPY license.html ./

# update timestamp
RUN touch ./src/lib.rs

CMD [ "/bin/bash", "-c", "maturin build \
    --release \
    --zig \
    --compatibility manylinux2014 \
    --interpreter python3.8 python3.9 python3.10 python3.11 python3.12 pypy3.8 pypy3.9 pypy3.10 \
    --color always" ]
