# vinbigdata-xray

docker で現在の位置で実行
```
# 例
docker run --rm -u `id -u`:`id -g` -v $HOME:$HOME --workdir $PWD --name="murakami-dev"  notify_simple_python_img bash -c "id -u && id -u && whoami"
```
