package gocudnn

import "log"

type keeper interface {
	keepsalive()
}

func keepsalivecheck(args []keeper) {
	for i := range args {
		if args[i] != nil {
			args[i].keepsalive()
		}

	}

}
func keepsalivebuffer(args ...interface{}) {
	keepers := make([]keeper, 0)
	for i := range args {
		switch y := args[i].(type) {
		case keeper:
			keepers = append(keepers, y)
		default:
			log.Printf("%T, put through keepsalive buffer. Make it a keeper interface or take it out of the buffer function", y)
		}

	}
	keepsalivecheck(keepers)
}
