# -*-coding=utf-8-*-
import json
import urllib.request
import urllib.error
import requests
import time

post_url = ''
key = ''
secret = ''
filepath = ''
http_path = ''




def main():
    req = urllib.request.Request(url=post_url, data=http_path)
    facial_feats = requests.post(url=post_url, data=http_path)
    req = requests.Request(post_url, http_path)
    facial_feats = requests.post(url=post_url, data=http_path)


if __name__ == "__main__":
    main()


