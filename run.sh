#! /bin/bash

src=""
dst=""
debug=""

function show_help {
    echo "usage:  $BASH_SOURCE --src <src> --dst <dst> --debug <debug>"
    echo "                     --src   : input path "
    echo "                     --dst   : output path "
    echo "                     --debug : is 0 or 1 "
}
function check_args {
    if [[ "$#" != 6 ]]; then
        echo "Error: missing arguments"
        show_help
        exit 1
    fi
}

# Read command line options
ARGUMENT_LIST=(
    "src"
    "dst"
    "debug"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)
check_args $@

eval set --$opts

while true; do
    case "$1" in
    h)
        show_help
        exit 0
        ;;
    -src|--src)  
        shift
        src=$1
        ;;
    -dst|--dst)  
        shift
        dst=$1
        ;;
    -debug|--debug)  
        shift
        debug=$1
        ;;
      --)
        shift
        break
        ;;
    esac
    shift
done

echo "src: $src"
echo "dst: $dst"
echo "debug: $debug"
