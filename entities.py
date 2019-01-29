import base64
import json
import zlib

MAP_VERSION = 68722819072

class Blueprint:

    def __init__(self):
        self.item = "blueprint"
        self.label = ""
        self.entities = {}
        self.icons = []
        self.version = MAP_VERSION

    def as_dict(self):
        d = {
            "icons": [{"signal": icon.as_dict(), "index": i + 1}
                      for i, icon in enumerate(self.icons)],
            "entities": [{**entity.as_dict(self), "entity_number": index}
                for entity, index in self.entities.items()],
            "item": self.item,
            "version": self.version
        }
        if self.label:
            d['label'] = self.label
        return d

    def as_exchange_str(self):
        jsondata = json.dumps({'blueprint': self.as_dict()})
        return '0' + base64.encodestring(zlib.compress(
            jsondata.encode('utf8'), 9)).decode('ascii')

    def get_entity_id(self, entity):
        return self.entities[entity]

    def add(self, obj):
        if isinstance(obj, Entity):
            self.entities[obj] = len(self.entities) + 1

class Direction:
    NORTH = 0
    EAST = 2
    SOUTH = 4
    WEST = 6

class Entity:

    def __init__(self, pos=(0, 0), dir=Direction.NORTH, **kw):
        self.position = Position(*pos)
        self.direction = dir

    def as_dict(self, blueprint):
        d = {
            "name": self.name,
            "position": self.position.as_dict(),
        }
        if self.direction != Direction.NORTH:
            d['direction'] = self.direction
        return d

class ConnectableMixin:

    has_two_connections = False

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.connections = Connection(self, self.has_two_connections)

    def as_dict(self, blueprint):
        if self.connections.is_empty():
            return super().as_dict(blueprint)
        return {**super().as_dict(blueprint),
            "connections": self.connections.as_dict(blueprint),
        }

class ControlBehaviorMixin:

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.control_behavior = self.behavior_cls(**kwargs)

    def as_dict(self, blueprint):
        if self.control_behavior.is_empty():
            return super().as_dict(blueprint)
        return {**super().as_dict(blueprint),
            "control_behavior": self.control_behavior.as_dict()
        }

class ControlBehavior:

    def is_empty(self):
        return False

    def as_dict(self):
        return {
            self.type: self.as_dict0()
        }

class DeciderBehavior(ControlBehavior):
    type = "decider_conditions"

    def __init__(self, left=None, cmp="<", right=0, out=None,
                 cp_count=True, **kw):
        self.left = left
        self.comparator = cmp
        self.right = right
        self.output = out
        self.copy_count = cp_count

    def as_dict0(self):
        d = {'comparator': self.comparator,
             'copy_count_from_input': self.copy_count}
        if self.left:
            d['first_signal'] = self.left.as_dict()
        if type(self.right) == int:
            d['constant'] = self.right
        elif self.right is not None:
            d['second_signal'] = self.right.as_dict()
        if self.output:
            d['output_signal'] = self.output.as_dict()
        return d

class ArithmeticBehavior(ControlBehavior):
    type = "arithmetic_conditions"

    def __init__(self, left=None, op="*", right=0, out=None, **kw):
        self.left = left
        self.operation = op
        self.right = right
        self.output = out

    def as_dict0(self):
        d = {'operation': self.operation}
        if type(self.left) == int:
            d['first_constant'] = self.left
        elif self.left is not None:
            d['first_signal'] = self.left.as_dict()
        if type(self.right) == int:
            d['second_constant'] = self.right
        elif self.right is not None:
            d['second_signal'] = self.right.as_dict()
        if self.output:
            d['output_signal'] = self.output.as_dict()
        return d

class ConstantBehavior(ControlBehavior):

    MAX_FILTER = 18

    def __init__(self, is_on=True, **kw):
        self.filters = {}
        self.is_on = is_on

    def is_empty(self):
        return self.is_on and not self.filters

    def add_signal(self, signal, count=1, index=None):
        if index is None:
            index = 1
            while index in self.filters:
                index += 1
        assert index >= 1 and index <= self.MAX_FILTER
        self.filters[index] = (signal, count)

    def as_dict(self):
        d = {}
        if not self.is_on:
            d['is_on'] = False
        if self.filters:
            d['filters'] = [{
                    "signal": signal.as_dict(),
                    "count": count,
                    "index": index
                } for (index, (signal, count)) in self.filters.items()]
        return d

class Signal:

    def __init__(self, type, name):
        self.type = type
        self.name = name

    def as_dict(self):
        return {
            "type": self.type,
            "name": self.name
        }

class Position:

    def __init__(self, x=0.0, y=0.0):
        self.x = float(x)
        self.y = float(y)

    def as_dict(self):
        return {
            "x": self.x,
            "y": self.y
        }

class Connection:

    def __init__(self, entity_handle, has_second):
        self.first = ConnectionPoint(entity_handle, 1 if has_second else None)
        self.second = ConnectionPoint(entity_handle, 2) if has_second else None

    def is_empty(self):
        return self.first.is_empty() and (
            not self.second or self.second.is_empty())

    def as_dict(self, blueprint):
        d = {}
        if not self.first.is_empty():
            d['1'] = self.first.as_dict(blueprint)
        if self.second and not self.second.is_empty():
            d['2'] = self.second.as_dict(blueprint)
        return d

class ConnectionPoint:

    def __init__(self, entity_handle, circuit_id):
        self.reds = []
        self.greens = []
        self.entity_handle = entity_handle
        self.circuit_id = circuit_id

    def join(self, other, color):
        if color == 'green':
            self_ch, other_ch = self.greens, other.greens
        elif color == 'red':
            self_ch, other_ch = self.reds, other.reds
        else:
            assert False
        self_ch.append(ConnectionData(other.entity_handle, other.circuit_id))
        other_ch.append(ConnectionData(self.entity_handle, self.circuit_id))
        

    def is_empty(self):
        return not self.reds and not self.greens

    def as_dict(self, blueprint):
        d = {}
        if self.reds:
            d['red'] = [con.as_dict(blueprint) for con in self.reds]
        if self.greens:
            d['green'] = [con.as_dict(blueprint) for con in self.greens]
        return d

class ConnectionData:

    def __init__(self, entity_handle, circuit_id=None):
        self.entity_handle = entity_handle
        self.circuit_id = circuit_id

    def as_dict(self, blueprint):
        d = {
            "entity_id": blueprint.get_entity_id(self.entity_handle),
        }
        if self.circuit_id:
            d['circuit_id'] = self.circuit_id
        return d

class IOCombinator(ControlBehaviorMixin, ConnectableMixin, Entity):
    has_two_connections = True
    input = property(lambda self: self.connections.first)
    output = property(lambda self: self.connections.second)

class DeciderCombinator(IOCombinator):
    behavior_cls = DeciderBehavior
    name = "decider-combinator"

class ArithmeticCombinator(IOCombinator):
    behavior_cls = ArithmeticBehavior
    name = "arithmetic-combinator"

class ConstantCombinator(ControlBehaviorMixin, ConnectableMixin, Entity):
    behavior_cls = ConstantBehavior
    name = "constant-combinator"
    output = property(lambda self: self.connections.first)

    def add_signal(self, *args, **kwargs):
        self.control_behavior.add_signal(*args, **kwargs)

class ElectricPole(ConnectableMixin, Entity):
    def __init__(self, type="medium-electric-pole", **kwargs):
        self.name = type
        super().__init__(**kwargs)

class SignalDefs(type):
    def __getattr__(self, attr):
        val = None
        if len(attr) == 1:
            if attr >= 'A' and attr <= 'Z':
                val = Signal("virtual", "signal-" + attr)
        elif len(attr) == 2 and attr[0] == 'N':
            if attr[1] >= '0' and attr[1] <= '9':
                val = Signal("virtual", "signal-" + attr[1])
        elif attr in ["RED", "GREEN", "BLUE", "YELLOW", "PINK",
                      "CYAN", "WHITE", "GREY", "BLACK"]:
            val = Signal("virtual", "signal-" + attr.lower())
        elif attr == 'EVERY':
            val = Signal("virtual", "signal-everything")
        if not val:
            raise AttributeError(attr)
        setattr(self, attr, val)
        return val

class Signals(metaclass=SignalDefs):
    pass
    
