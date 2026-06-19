resolve_experiment() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: experiment required" >&2
    return 1
  fi
  case "$1" in
    mnist|cifar)
		module="experiments.images.${1}.${1}"
		ARGS=(--epochs 50 --batch-size 256 --t-max 1.0 --seed 42 --runs 5)
		;;
    *)
      echo "Unknown experiment: $1" >&2
      return 1
      ;;
  esac
}