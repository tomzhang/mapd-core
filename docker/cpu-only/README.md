## Docker Install

Read through everything before blindly running the commands below.

Install Docker:
- https://docs.docker.com/engine/installation/
  - Install from Docker's repos, not your distro's. Normal distros are often many versions behind.
  - Don't bother installing from Homebrew on Mac - use Docker's installer so everything works correctly with xhyve.
    - Might require you to upgrade or remove VirtualBox.

## Building MapD container

The `Dockerfile` assumes that there is a MapD tarball sitting in the same directory named `mapd2-latest-Linux-x86_64-nocuda.tar.gz`.

To build the container do:

    wget --ask-password https://mapd@builds.mapd.com/mapd2-latest-Linux-x86_64-nocuda.tar.gz # use your own user/pass
    docker build .

The image id will be output on the last line of the `build` step. To assign a custom name do something like:

    docker build -t mapd/mapd-nocuda:v1.2.5

which will assign the name `mapd/mapd-nocuda` and the tag `v1.2.5` to the image.

### Image layout

The tarball gets extracted to `/installs`. When building the `Dockerfile`, the extracted tarball then gets symlinked to `/mapd`.

Data directory lives at `/mapd-storage/data`.

Config file lives at `/mapd-storage/mapd.conf`.

## Starting MapD inside a container

    docker run -p 19092:9092 mapd/mapd-nocuda:v1.2.5

That will expose the webserver on port `19092`.

Saved data inside containers is ephemeral. To preserve your data you probably want to use a data container or at least bind mount in a host directory.

    docker run -v /home/mapd/prod/mapd-storage:/mapd-storage -p 19092:9092 mapd/mapd-norender:v1.1.9

will mount the host directory `/home/mapd/prod/mapd-storage` to `/mapd-storage` in the container.

See the Docker docs for more info on how to run as a daemon, how to spawn a shell inside the container, how to autostart on reboot, etc.

Note: the `Dockerfile` currently uses `startmapd` to start both `mapd_web_server` and `mapd_server`. Make sure you have run `initdb` on your data folder before running (it usually gets run when you build).

## Ignore all of the above and do this

Download and install Docker. For Mac see https://docs.docker.com/docker-for-mac/

Open a terminal and verify that Docker is installed:

    docker version

Go to the path containing the `Dockerfile` next to this readme and download the latest `nocuda` *Linux* build of MapD:

    cd /path/to/mapd2/docker/cpu-only
    wget --ask-password https://mapd@builds.mapd.com/mapd2-latest-Linux-x86_64-nocuda.tar.gz

Build the container and give it a tag:

    docker build -t mapd/mapd-nocuda:latest .

This will take some time as it also downloads the base image and installs a few dependencies.

Try to run with:

    docker run -d -p 19092:9092 mapd/mapd-nocuda:latest

This should echo out a long random string. This string is your `[container id]`.

You should now be able to hit http://localhost:19092/v2

Saved data is ephemeral. Copy the generated `mapd-storage` to somewhere on your machine.

    # copy the /mapd-storage from the container you just started to a local dir
    docker cp [container id]:/mapd-storage .

The `-d` option in the `docker run -d` above tells docker to run the container in the background. Stop this container before starting another instance.

    # grab the container id of your running container if you forgot it
    docker ps
    # stop the container
    docker stop [container id]

### Typical usage

The above was only required to get the initial container set up. For day-to-day development do the following:

    docker run -d -p 19092:9092 -v /path/to/local/mapd-storage:/mapd-storage -v /path/to/frontend/public:/mapd/frontend mapd/mapd-nocuda:latest

Replace `/path/to/local/mapd-storage` with the full path to the `mapd-storage` dir you copied out initially.
Replace `/path/to/frontend/public` with the `public`/`dist`/`frontend` dir of the dir you're building. Leave out this portion if you just want to use the frontend bundled with the server.

Do stop the backend, run:

    # grab the container id of your running container
    docker ps
    # stop the container
    docker stop [container id]
