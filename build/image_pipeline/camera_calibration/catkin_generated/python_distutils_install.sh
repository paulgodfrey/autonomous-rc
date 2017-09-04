#!/bin/sh

if [ -n "$DESTDIR" ] ; then
    case $DESTDIR in
        /*) # ok
            ;;
        *)
            /bin/echo "DESTDIR argument must be absolute... "
            /bin/echo "otherwise python's distutils will bork things."
            exit 1
    esac
    DESTDIR_ARG="--root=$DESTDIR"
fi

echo_and_run() { echo "+ $@" ; "$@" ; }

echo_and_run cd "/home/nvidia/autonomous-rc/src/image_pipeline/camera_calibration"

# snsure that Python install destination exists
echo_and_run mkdir -p "$DESTDIR/home/nvidia/autonomous-rc/install/lib/python2.7/dist-packages"

# Note that PYTHONPATH is pulled from the environment to support installing
# into one location when some dependencies were installed in another
# location, #123.
echo_and_run /usr/bin/env \
    PYTHONPATH="/home/nvidia/autonomous-rc/install/lib/python2.7/dist-packages:/home/nvidia/autonomous-rc/build/lib/python2.7/dist-packages:$PYTHONPATH" \
    CATKIN_BINARY_DIR="/home/nvidia/autonomous-rc/build" \
    "/usr/bin/python" \
    "/home/nvidia/autonomous-rc/src/image_pipeline/camera_calibration/setup.py" \
    build --build-base "/home/nvidia/autonomous-rc/build/image_pipeline/camera_calibration" \
    install \
    $DESTDIR_ARG \
    --install-layout=deb --prefix="/home/nvidia/autonomous-rc/install" --install-scripts="/home/nvidia/autonomous-rc/install/bin"
