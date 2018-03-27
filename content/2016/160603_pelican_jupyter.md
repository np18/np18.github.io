Title: Configure Pelican for Jupyter notebooks, code, and math display
Date: 2016-06-03 05:00
Category: 
Tags: python, website, github
Author: Eric Carlson
slug: pelican-jupyter-and-code
Series: pelican-website
Status: published

As a final step we'll configure Pelican to enable Jupyter notebook integration.  This will let us
quickly copy in analysis .ipynb files and include math, markdown, and figures.  In further configuration
we can even enable Pelican plugins for UML diagrams to add text-generated activity/state diagrams,
or can add plugins for comments to interact with readers.


Now add pelican plugins and themes and activate as needed...

	:::bash
	$ cd ~/code/ext
	$ git clone --recursive https://github.com/getpelican/pelican-plugins

Also add theme to start, using Jake Vanderplaas' since he wrote the notebook plugin I'm using, should work together...

	:::bash
	$ git submodule add https://github.com/jakevdp/pelican-octopress-theme.git themes/octopress

Add liquid plugin to pelicon conf and add theme (`pelicanconf.py`):

	PLUGIN_PATHS = ['/Users/ecarlson/code/external/pelican-plugins', ]
	PLUGINS = ['liquid_tags.img', 'liquid_tags.video',
			   'liquid_tags.youtube', 'liquid_tags.vimeo',
			   'liquid_tags.include_code', 'liquid_tags.notebook']

	NOTEBOOK_DIR = 'notebooks'
	
	EXTRA_HEADER = open('_nb_header.html').read().decode('utf-8') if os.path.exists('_nb_header.html') else None
	
	THEME = 'themes/octopress'
	
I also set to default articles as draft mode:

	DEFAULT_METADATA = {
		'status': 'draft',
	}

With these changes we should now be able to create Jupyter notebooks and include them in our blog
posts (tests below).

Lastly, I like to modify my jupyter to auto-save python files when saving .ipynb - this makes it
easier in git to tell what changed from version to version, as often meaninglss changes (e.g.
re-running a notebook, which changes cell numbering) result in a commit log noise.

`~/.jupyter/jupyter_notebook_config.py`:

	:::python
	import os
	from subprocess import check_call
	
	def post_save(model, os_path, contents_manager):
		"""post-save hook for converting notebooks to .py scripts"""
		if model['type'] != 'notebook':
			return # only do this for notebooks
		d, fname = os.path.split(os_path)
		check_call(['ipython', 'nbconvert', '--to', 'script', fname], cwd=d)
	
	c = get_config()
	c.FileContentsManager.post_save_hook = post_save

## Markup Test

Python code:

    #!python
    print("The path-less shebang syntax *will* show line numbers.")
    print('Testing 1 2 3')
    
Bash code:
	
	#!bash
	for n in t1 t2 t3; do
	  echo $n
	done


Notebook test:

{% notebook 161201_pelican_setup/test_notebook.ipynb %}


## References

* https://www.notionsandnotes.org/tech/web-development/pelican-static-blog-setup.html
* https://h-gens.github.io/getting-started-with-pelican-and-ipython-notebooks.html
* https://pages.github.com/
* https://github.com/getpelican/pelican-themes/tree/master/pelican-bootstrap3
* https://rjweiss.github.io/articles/2014_03_31/testing-ipython-integration/
* http://danielfrg.com/blog/2013/03/08/pelican-ipython-notebook-plugin/
* http://docs.getpelican.com/en/3.1.1/getting_started.html