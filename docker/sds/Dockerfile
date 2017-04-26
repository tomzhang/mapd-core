FROM ubuntu:16.04
MAINTAINER Andrew Seidl <andrew@mapd.com>

ADD mapd-latest-Linux-x86_64.tar.gz /installs
RUN rm -f /mapd && ln -s $(ls -dtr /installs/mapd* | head -n 1) /mapd
RUN mkdir -p /mapd-storage/sds

WORKDIR /mapd

EXPOSE 10301

CMD /mapd/bin/StringDictionaryServer --port 10301 --path /mapd-storage/sds
