from asciimatics.effects import Cycle, Stars
from asciimatics.renderers import FigletText
from asciimatics.scene import Scene
from asciimatics.screen import Screen
from time import sleep

def deft_splash(screen):
    screen.print_at('Hello world!', 0, 0)
    screen.refresh()
    sleep(10)

Screen.wrapper(demo)

def deft_testsplash(screen):
    effects = [
               Cycle(
                     screen,
                     FigletText("dEFT", font='big'),
                     int(screen.height / 2 - 8)),
               Cycle(
                     screen,
                     FigletText("A differential Effective Field Theory tool", font='small'),
                     int(screen.height / 2 + 3)),
               Stars(screen, 900)
               ]
    screen.play([Scene(effects, 900)])

#Screen.wrapper(deft_splash)
