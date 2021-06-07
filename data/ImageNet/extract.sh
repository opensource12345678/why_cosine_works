#!/bin/bash

STRIP=${1%.*}                                #strip last suffix
NAME=${STRIP%.tar}                           #strip .tar suffix, if present
tar -xf "$1" --xform="s|^|$NAME/|S"          #run command
