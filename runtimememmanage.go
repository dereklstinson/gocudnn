package gocudnn

type keeper interface {
	keepsalive()
}

func keepsalive(args ...keeper) {
	for i := range args {
		if args[i] != nil {
			args[i].keepsalive()
		}

	}

}
