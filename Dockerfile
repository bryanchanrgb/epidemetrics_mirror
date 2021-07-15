FROM python:3.9.5-slim

WORKDIR /

## Now install R and littler, and create a link for littler in /usr/local/bin
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libcurl4-openssl-dev libnlopt-dev littler \
        r-cran-littler r-base r-base-dev r-base-core r-recommended \
		r-cran-devtools r-cran-readr r-cran-ggplot2 r-cran-gridextra  \
        r-cran-plyr r-cran-dplyr r-cran-ggsci r-cran-rcolorbrewer \
        r-cran-viridis r-cran-sf r-cran-reshape2 \
        r-cran-scales r-cran-plotrix \
	&& ln -s /usr/lib/R/site-library/littler/examples/install.r /usr/local/bin/install.r \
	&& ln -s /usr/lib/R/site-library/littler/examples/install2.r /usr/local/bin/install2.r \
	&& ln -s /usr/lib/R/site-library/littler/examples/installBioc.r /usr/local/bin/installBioc.r \
	&& ln -s /usr/lib/R/site-library/littler/examples/installDeps.r /usr/local/bin/installDeps.r \
	&& ln -s /usr/lib/R/site-library/littler/examples/installGithub.r /usr/local/bin/installGithub.r \
	&& ln -s /usr/lib/R/site-library/littler/examples/testInstalled.r /usr/local/bin/testInstalled.r \
	&& pip install --upgrade pip setuptools \
    && install.r --error docopt \
    && install2.r --error readr ggplot2 gridExtra plyr dplyr  ggsci viridis reshape2 scales plotrix ggallin \
    && R -e 'devtools::install_github("kassambara/ggpubr")' \
    && rm -rf /tmp/downloaded_packages/ /tmp/*.rds \
    && rm -rf /var/lib/apt/lists/*

RUN R -e 'install.packages("egg")'
COPY requirements.txt .
RUN pip3 install -r requirements.txt

# copy the content of the local src directory to the working directory
COPY src /src
COPY web /web
COPY tests /tests
COPY R /R

COPY start.sh /start.sh
COPY run_tests.sh /run_tests.sh

CMD [ "/bin/sh", "/start.sh" ]