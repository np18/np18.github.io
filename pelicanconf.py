#!/usr/bin/env python
# -*- coding: utf-8 -*- #
from __future__ import unicode_literals
import os 

AUTHOR = 'Nidhin Pattaniyil'
SITENAME = 'NP Blog'
SITEURL = ''

PATH = 'content'

TIMEZONE = 'EST'

DEFAULT_LANG = 'en'

# Feed generation is usually not desired when developing
FEED_ALL_ATOM = None
CATEGORY_FEED_ATOM = None
TRANSLATION_FEED_ATOM = None
AUTHOR_FEED_ATOM = None
AUTHOR_FEED_RSS = None

# Blogroll
LINKS = None

# Social widget
SOCIAL = (
             ('LinkedIn', 'https://www.linkedin.com/in/erictcarlson'),
             ('Github', 'https://github.com/npatta01'),
         )
         

ARTICLE_SAVE_AS = '{date:%Y}/{slug}.html'
ARTICLE_URL = '{date:%Y}/{slug}.html'

DEFAULT_PAGINATION = 10

# Uncomment following line if you want document-relative URLs when developing
RELATIVE_URLS = True

#THEME = "/Users/nidhin.pattaniyil/demo/blog/pelican-themes/mnmlist"
THEME = "themes/pelican-bootstrap3"

#JINJA_ENVIRONMENT = {'extensions': ['jinja2.ext.i18n']}
BOOTSTRAP_THEME ="yeti"
PLUGIN_PATHS = ['plugins']
#PLUGINS = ['assets', 'sitemap', 'gravatar']

PLUGINS = ['liquid_tags.img', 'liquid_tags.video',
           'liquid_tags.youtube', 'liquid_tags.vimeo',
           'liquid_tags.include_code', 'liquid_tags.notebook',
           'i18n_subsites','assets', 'sitemap', 'gravatar','assets','tag_cloud'
           ]


JINJA_ENVIRONMENT = {
    'extensions': ['jinja2.ext.i18n'],
}

DISPLAY_CATEGORIES_ON_MENU = False
DISPLAY_RECENT_POSTS_ON_SIDEBAR = True
DISPLAY_CATEGORIES_ON_SIDEBAR = True

STATIC_PATHS = ['images', 'extra']
EXTRA_PATH_METADATA = {'extra/CNAME': {'path': 'CNAME'},}



TAG_CLOUD_STEPS = 1

NOTEBOOK_DIR = 'notebooks'

EXTRA_HEADER = open('_nb_header.html').read() if os.path.exists('_nb_header.html') else None


