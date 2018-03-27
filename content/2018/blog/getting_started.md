Title: How To Blog with Pelican
Date: 2018-03-27
Category: tutorial
Tags: blog , python, tutorial

In this tutotial, we will cover creating a blog with Pelican and host it on Github  

The benefits of blogging:
- personl brand 
- share your knowledge
- helps like minded people connect with you.

At work, we use python, so I choose [Pelican]().


# Setting up Env

Create a python Env 
```
conda create -n blog python=3.6
```

Setting up Pelican
```
pip install pelican markdown
pelican-quickstart
```

These are the options I generated
Pelican Options
```

```

# Pelican File 
The main files are pelican.conf and publishconf.py.

This is my pelican.conf

This is my publish.conf

# Customizing Pelican 
Pelican has a lot of plugins and themes.
```
git submodule add https://github.com/getpelican/pelican-plugins plugins
git clone --recursive https://github.com/getpelican/pelican-themes themes


```

**Running Locally**
```
make devserver
```

**Creating Blogpost**    
Created a blogpost is as simple as creating a md file in the content folder .
A simple markdown file looks like
```

```

# Folder Structure 



# Deploying to Github
Github is a nice solution to host sites like blogs.
Create a repo {github username}.github.io. 
Github will render you master branch and be available at gitusername.github.io

To deply using pelican
```
USERNAME=np18
make publish
ghp-import output
git push git@github.com:${USERNAME}/${USERNAME}.github.io.git 

pelican content -o output -s pelicanconf.py

make publish
ghp-import output
git push git@github.com:${USERNAME}/${USERNAME}.github.io.git  gh-pages:master --force

```


After a while, this setup got tiring. 
So, I now use continous integration with [TravisCI]()

** Deploy using Travis CI
My Travis file looks like 
```
```

I needed to create a git personal access token and grant it "read_public_scope" at github.
At Travis, I needed to create an environment variable "" .

Instead of commiting to the master branch, I develop in the source branch.
My CI is the one responsible for pushing to the master branch.


# Using Own Domain 
DNS Info 
[a link relative to the current file]({filename}dns.png)      

[a link relative to the content root]({filename}/dns.png)  

![Icon]({attach}dns.png)
![Photo]({attach}/dns.jpg)