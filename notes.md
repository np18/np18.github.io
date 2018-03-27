


```
USERNAME=np18
make publish
ghp-import output
git push git@github.com:${USERNAME}/${USERNAME}.github.io.git 

pelican content -o output -s pelicanconf.py

make publish
ghp-import output
git push git@github.com:${USERNAME}/${USERNAME}.github.io.git  gh-pages:master --force


https://github.com/getpelican/pelican-themes/tree/master/pelican-bootstrap3
```            


```
USERNAME=np18
pelican content -o output -s pelicanconf.py
ghp-import output

git@github.com:np18/np18.github.io.git

git push git@github.com:${USERNAME}/${USERNAME}.github.io.git gh-pages:master --force 


git push git@github.com:${USERNAME}/${USERNAME}.github.io.git master --force 

```



```
git clone --recursive https://github.com/getpelican/pelican-themes ~/pelican-themes
git clone --recursive https://github.com/getpelican/pelican-plugins ~/pelican-plugins

```

pip install jupyter