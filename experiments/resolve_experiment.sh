resolve_experiment() {
  if [[ $# -lt 1 ]]; then
    echo "Usage: experiment required" >&2
    return 1
  fi
  case "$1" in
    mnist|cifar)
		module="experiments.images.run"
		ARGS=(--dataset "$1" --epochs 50 --batch-size 256 --t-max 1.0 --seed 42 --runs 5)
    lambdas=(0.0001 0.001 0.01)
		;;
    g[1-9])
		module="experiments.annuli.annuli"
		ARGS=(--dim "${1#g}" --epochs 100 --batch-size 256 --t-max 1.0 --seed 42 --runs 10)
    lambdas=(0.0001 0.001 0.01 0.1)
		;;
    *)
      echo "Unknown experiment: $1" >&2
      return 1
      ;;
  esac
}