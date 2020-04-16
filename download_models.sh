#!/bin/bash

# If this script fails, try the following links in your browser:
#https://drive.google.com/file/d/1bqQ2GgwyINrqjz2y9Ob_O9URZRVA5DcD/view?usp=sharing
#https://drive.google.com/file/d/1KoEZo3x4z7S5pXlsNYLu2QHnGG9npDhH/view?usp=sharing

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1bqQ2GgwyINrqjz2y9Ob_O9URZRVA5DcD" > /tmp/tmp.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/tmp.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > multires.pb

curl -c /tmp/cookies "https://drive.google.com/uc?export=download&id=1KoEZo3x4z7S5pXlsNYLu2QHnGG9npDhH" > /tmp/tmp.html
curl -L -b /tmp/cookies "https://drive.google.com$(cat /tmp/tmp.html | grep -Po 'uc-download-link" [^>]* href="\K[^"]*' | sed 's/\&amp;/\&/g')" > vanilla.pb

