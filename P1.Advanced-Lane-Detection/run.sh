#! /bin/bash

src=""
dst=""
debug="0"

function show_help {
    echo "usage:  $BASH_SOURCE --src <src> --dst <dst> --debug <debug>"
    echo "                     --src   : input path "
    echo "                     --dst   : output path "
    echo "                     --debug : is 0 or 1 "
    echo "or:     $BASH_SOURCE --src <src> --dst <dst>"
    echo "                     --src   : input path "
    echo "                     --dst   : output path "
}

function check_args {
    if [[ "$#" != 6 ]] && [[ "$#" != 4 ]]; then
        show_help
        exit 1
    fi
}

# Read command line options
ARGUMENT_LIST=(
    "src"
    "dst"
    "Debug"
)

# read arguments
opts=$(getopt \
    --longoptions "$(printf "%s:," "${ARGUMENT_LIST[@]}")" \
    --alternative \
    --name "$(basename "$0")" \
    --options "" \
    -- "$@"
)

check_args $@

eval set --$opts

while [[ $# -gt 0 ]]; do
    case "$1" in
    --src)  
        src=$2
        shift 2
        ;;
    --dst)  
        dst=$2
        shift 2
        ;;
    --Debug)  
        debug=$2
        shift 2
        ;;
    --)
        shift 1
        ;;
    *)
        show_help
        exit 1
        break
        ;;
    esac
done

echo "Src: $src";
echo "Dst: $dst";
echo "Debug Mode: $debug";
python3  phase1_advanced_lane_detection.py $src $dst $debug

