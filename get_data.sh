if [ -d data]; then
    echo "data directory already exists"
    else mkdir data
fi

if [ -d data/ICFHR_package ]; then
    echo "data already exists, cleaning up the directory"
    rm -rf data/ICFHR_package
fi

if [ -f data/ICFHR_package.zip ]; then
    echo "Would you like to redownload?"
    select yn in "Yes" "No"; do
        case $yn in
            Yes )
              rm data/ICFHR_package.zip;
              wget http://www.isical.ac.in/~crohme/ICFHR_package.zip -P data;
              break;;
            No )
              echo "Using the existing zip file";;
        esac
    done
    else wget http://www.isical.ac.in/~crohme/ICFHR_package.zip -P data
fi
unzip data/ICFHR_package.zip -d data && rm data/ICFHR_package.zip
