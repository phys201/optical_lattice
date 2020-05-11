from jug import value, set_jugdir
import jugfile
set_jugdir('jugfile.jugdata')


# Demonstrate we can retrieve results from jug data
results = value(jugfile.fullresults)
settings = value(jugfile.settings)