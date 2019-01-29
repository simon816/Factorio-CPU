from entities import *

class Model:

    def __init__(self, *entities):
        self.entities = entities
        self.inputs = {}
        self.outputs = {}

    def define_input(self, name, conn, channels):
        self.inputs[name] = (conn, channels)

    def define_output(self, name, conn, channels):
        self.outputs[name] = (conn, channels)

    def connect_input(self, name, conn, channel):
        in_conn, allow_ch = self.inputs[name]
        assert channel in allow_ch
        in_conn.join(conn, channel)

    def connect_output(self, name, conn, channel):
        out_conn, allow_ch = self.outputs[name]
        assert channel in allow_ch
        out_conn.join(conn, channel)

    def get_input(self, name):
        return self.inputs[name][0]

    def get_output(self, name):
        return self.outputs[name][0]

    def shift(self, x, y):
        for entity in self.entities:
            entity.position.x += x
            entity.position.y += y

    def add_to_blueprint(self, blueprint):
        for entity in self.entities:
            blueprint.add(entity)

    @staticmethod
    def compose(first, second, *rest):
        entities = []
        entities.extend(first.entities)
        entities.extend(second.entities)
        for m in rest:
            entities.extend(m.entities)
        model = Model(*entities)
        return model

def make_oneshot_pulse(pulse_clock=Signals.C, output=Signals.D):
    c1 = DeciderCombinator(left=pulse_clock, cmp=">", right=0, out=output,
                      cp_count=False, dir=Direction.EAST)
    c2 = ArithmeticCombinator(left=output, op="*", right=-1, out=output,
                              pos=(0, 1), dir=Direction.EAST)
    c1.output.join(c2.input, 'red')
    c1.output.join(c2.output, 'green')
    model = Model(c1, c2)
    model.define_input("trigger", c1.input, {'red', 'green'})
    model.define_output("pulse", c1.output, {'green'})
    return model

def make_rom_select(size, PC):
    prev = None
    entities = []
    for addr in range(1, size+1):
        c1 = ConstantCombinator(dir=Direction.EAST,  pos=(0, addr-1))
        c2 = DeciderCombinator(left=PC, cmp="=", right=addr, out=Signals.EVERY,
                          dir=Direction.EAST, pos=(1.5, addr-1))
        c1.output.join(c2.input, 'green')
        if prev:
            c2.input.join(prev.input, 'red')
            c2.output.join(prev.output, 'red')
        entities.append(c1)
        entities.append(c2)
        prev = c2
    model = Model(*entities)
    first = entities[1]
    model.define_input("pc", first.input, {'red'})
    model.define_output("insn", first.output, {'red'})
    return model

def make_register(reg, write_latch=Signals.C, fast_write_latch=Signals.O):
    c_set = DeciderCombinator(left=write_latch, cmp=">", right=0, out=reg,
                              dir=Direction.EAST)
    c_sust = DeciderCombinator(left=write_latch, cmp="=", right=0, out=reg,
                               pos=(2, 0), dir=Direction.WEST)
    c_fast_write = DeciderCombinator(left=fast_write_latch, cmp="≠", right=0,
                                     out=write_latch, cp_count=False,
                                     pos=(4, 0), dir=Direction.WEST)
    c_set.output.join(c_sust.output, 'green')
    c_set.input.join(c_sust.input, 'green')
    c_sust.input.join(c_fast_write.output, 'green')
    c_sust.input.join(c_sust.output, 'red')
    c_sust.output.join(c_set.output, 'red')
    model = Model(c_set, c_sust, c_fast_write)
    model.define_output("data", c_set.output, {'green'})
    model.define_input("d_write", c_set.input, {'red'})
    model.define_input("clock", c_sust.input, {'green'})
    model.define_input("fast_write", c_fast_write.input, {'green', 'red'})
    return model

def make_program_counter(PC, clock_sig=Signals.C, op_signal=Signals.O):
    one = ConstantCombinator(dir=Direction.EAST)
    one.add_signal(PC, count=1)
    data_latch = DeciderCombinator(left=op_signal, cmp=">", right=0,
                                   out=PC, dir=Direction.EAST, pos=(1.5, 0))
    write_latch = DeciderCombinator(left=op_signal, cmp=">", right=0,
                                    out=clock_sig, cp_count=False,
                                    dir=Direction.EAST, pos=(1.5, 1))
    c_set = DeciderCombinator(left=clock_sig, cmp=">", right=0, out=PC,
                              dir=Direction.EAST, pos=(3.5, 0))
    c_sust = DeciderCombinator(left=clock_sig, cmp="=", right=0, out=PC,
                               dir=Direction.EAST, pos=(3.5, 1))
    one.output.join(data_latch.input, 'green')
    data_latch.input.join(write_latch.input, 'red')
    data_latch.output.join(c_set.input, 'red')
    write_latch.output.join(c_sust.input, 'green')
    c_sust.input.join(c_set.input, 'green')
    c_set.output.join(c_sust.output, 'red')
    c_sust.output.join(c_sust.input, 'red')
    model = Model(one, data_latch, write_latch, c_set, c_sust)
    model.define_input("fast_inc", data_latch.input, {'red'})
    model.define_input("pc_set", c_set.input, {'red'})
    model.define_input("clock", c_sust.input, {'green'})
    model.define_output("pc", c_set.output, {'red'})
    return model

def make_disable_inc_pc(opcode, PC, op_signal=Signals.O):
    plus1 = DeciderCombinator(left=op_signal, cmp="=", right=opcode, out=PC,
                      cp_count=False, dir=Direction.EAST)
    emitPC = DeciderCombinator(left=op_signal, cmp="=", right=opcode, out=PC,
                              dir=Direction.EAST, pos=(2, 0))
    negate = ArithmeticCombinator(left=PC, op="*", right=-1, out=PC,
                                  dir=Direction.EAST, pos=(4, 0))
    plus1.output.join(negate.input, 'red')
    emitPC.output.join(negate.input, 'red')
    plus1.input.join(emitPC.input, 'red')
    model = Model(plus1, emitPC, negate)
    model.define_input("insn", plus1.input, {'red'})
    model.define_output("pc", negate.output, {'red', 'green'})
    return model

def make_ram(size, addr_sig=Signals.A, data_sig=Signals.D):
    clock_sig = Signals.C
    height = 0
    entities = []
    head = None, None
    prev = None, None
    for addr in range(1, size+1):
        c_set = DeciderCombinator(left=clock_sig, cmp=">", right=0,
                                  out=data_sig, dir=Direction.EAST,
                                  pos=(2, height - 1))
        c_sust = DeciderCombinator(left=clock_sig, cmp="=", right=0,
                                   out=data_sig, dir=Direction.EAST,
                                   pos=(2, height))
        write_latch = DeciderCombinator(left=addr_sig, cmp="=", right=addr,
                                        out=Signals.EVERY, dir=Direction.NORTH,
                                        pos=(0, height))
        read_latch = DeciderCombinator(left=addr_sig, cmp="=", right=addr,
                                       out=data_sig, dir=Direction.SOUTH,
                                       pos=(3, height))
        entities.extend((c_set, c_sust, write_latch, read_latch))
        c_set.input.join(c_sust.input, 'green')
        c_set.output.join(c_sust.output, 'red')
        c_sust.output.join(c_sust.input, 'red')
        write_latch.output.join(c_set.input, 'green')
        read_latch.input.join(c_set.output, 'red')
        p_write, p_read = prev
        if p_write is not None:
            p_write.input.join(write_latch.input, 'green')
            p_read.input.join(read_latch.input, 'green')
            p_read.output.join(read_latch.output, 'red')
        else:
            head = write_latch, read_latch
        prev = write_latch, read_latch
        height -= 2

    write_gate = DeciderCombinator(left=addr_sig, cmp="≠", right=0,
                                   out=clock_sig, cp_count=False, pos=(0, 2))
    write_success = DeciderCombinator(left=clock_sig, cmp="≠", right=0,
                                      out=Signals.GREEN, cp_count=False,
                                      dir=Direction.SOUTH, pos=(1, 2))
    read_success = DeciderCombinator(left=addr_sig, cmp="≠", right=0,
                                     out=Signals.GREEN, cp_count=False,
                                     dir=Direction.SOUTH, pos=(3, 2))
    entities.extend((write_gate, write_success, read_success))
    write_gate.output.join(write_success.input, 'green')
    write_gate.output.join(head[0].input, 'green')
    write_gate.input.join(head[0].input, 'green')
    #write_success.output.join(write_gate.input, 'green')
    read_success.input.join(head[1].input, 'green')
    read_success.output.join(head[1].output, 'red')
    model = Model(*entities)
    model.define_input("read_addr", read_success.input, {'green'})
    model.define_input("write", write_gate.input, {'green'})
    model.define_output("data", read_success.output, {'red'})
    model.define_output("write_fin", write_success.output, {'green', 'red'})
    return model

class Params:
    FastWrite = lambda reg: ('fast_write', reg)
    DISABLE_AUTO_INC_PC = ('disable_inc_pc', None)
    SlowWrite = lambda reg: ('slow_write', reg)
    WRITE_PC = ('write_pc', None)
    RAM_READ = ('ram_read', None)
    RAM_WRITE = ('ram_write', None)

    @staticmethod
    def all(key, params):
       return {v for (k, v) in params if k == key}

# Add literal to reg, store in reg
def factory_OP_ADDL_REG(reg):
    def make_OP_ADDL_REG(regs, operands):
        c = ArithmeticCombinator(left=regs[reg], op="+", right=operands[0],
                             out=regs[reg], dir=Direction.EAST)
        model = Model(c)
        model.define_input("state_in", c.input, {'green', 'red'})
        model.define_output("reg_" + reg, c.output, {'green', 'red'})
        return model
    return make_OP_ADDL_REG

# Add reg to reg, store in reg
def factory_OP_ADD_REG(left, right, dest):
    def make_OP_ADD_REG(regs, operands):
        c = ArithmeticCombinator(left=regs[left], op="+", right=regs[right],
                             out=regs[dest], dir=Direction.EAST)
        model = Model(c)
        model.define_input("state_in", c.input, {'green', 'red'})
        model.define_output("reg_" + dest, c.output, {'green', 'red'})
        return model
    return make_OP_ADD_REG

def make_abs_jump(regs, operands):
    copy_val = ArithmeticCombinator(left=operands[0], op="+", right=0,
                                    out=Signals.P, dir=Direction.EAST)
    trigger_write = DeciderCombinator(left=Signals.O, cmp="≠", right=0,
                                      out=Signals.C, cp_count=False,
                                      dir=Direction.EAST, pos=(2, 0))
    copy_val.input.join(trigger_write.input, 'green')
    model = Model(copy_val, trigger_write)
    model.define_input("state_in", copy_val.input, {'green'})
    model.define_output("pc", copy_val.output, {'green', 'red'})
    model.define_output("pc_clock", trigger_write.output, {'green', 'red'})
    return model

# Load absolute address into A
def make_OP_LOAD_ABS_A(regs, operands):
    copy_val = ArithmeticCombinator(left=operands[0], op="+", right=0,
                                    out=Signals.A, dir=Direction.EAST)
    clock_ready = DeciderCombinator(left=Signals.GREEN, cmp="≠", right=0,
                                    out=Signals.C, cp_count=False,
                                    dir=Direction.WEST, pos=(4, 0))
    copy_data = ArithmeticCombinator(left=Signals.D, op="+", right=0,
                                     out=regs['A'], dir=Direction.WEST,
                                     pos=(2, 0))
    clock_ready.input.join(copy_data.input, 'red')
    model = Model(copy_val, clock_ready, copy_data)
    model.define_output("ram_addr", copy_val.output, {'green', 'red'})
    model.define_input("ram_res", clock_ready.input, {'red'})
    model.define_output("reg_A", copy_data.output, {'green', 'red'})
    model.define_output("reg_A_clock", clock_ready.output, {'green'})
    model.define_input("state_in", copy_val.input, {'green', 'red'})
    return model

# Store literal into absolute address
def make_OP_STOREL_ABS(regs, operands):
    copy_addr = ArithmeticCombinator(left=operands[0], op="+", right=0,
                                     out=Signals.A, dir=Direction.EAST)
    copy_data = ArithmeticCombinator(left=operands[1], op="+", right=0,
                                     out=Signals.D, dir=Direction.EAST,
                                     pos=(2, 0))
    clock_ready = DeciderCombinator(left=Signals.GREEN, cmp="≠", right=0,
                                    out=Signals.C, cp_count=False, pos=(4, 0),
                                    dir=Direction.WEST)
    copy_addr.input.join(copy_data.input, 'green')
    copy_addr.output.join(copy_data.output, 'green')
    model = Model(copy_addr, copy_data, clock_ready)
    model.define_input("state_in", copy_addr.input, {'green'})
    model.define_output("ram_data", copy_data.output, {'green'})
    model.define_input("ram_write_fin", clock_ready.input, {'green'})
    return model

opcodes = {
    1: {
        'params': {Params.FastWrite('A')},
        'make': factory_OP_ADDL_REG('A')
    },
    2: {
        'params': {Params.FastWrite('B')},
        'make': factory_OP_ADDL_REG('B')
    },
    3: {
        'params': {Params.FastWrite('A')},
        'make': factory_OP_ADD_REG('A', 'B', 'A')
    },
    4: {
        'params': {Params.FastWrite('B')},
        'make': factory_OP_ADD_REG('A', 'B', 'B')
    },
    5: {
        'params': {Params.DISABLE_AUTO_INC_PC, Params.WRITE_PC},
        'make': make_abs_jump
    },
    6: {
        'params': {Params.SlowWrite('A'), Params.RAM_READ},
        'make': make_OP_LOAD_ABS_A,
    },
    7: {
        'params': {Params.RAM_WRITE},
        'make': make_OP_STOREL_ABS
    }
}

def build_opcodes(reg_models, pc, ram, regs, operands, op_signal=Signals.O):
    entities = []
    height = 0
    fast_write_lines = {}
    reg_write_lines = {}
    prev_op_sel = head = None
    for opcode in sorted(opcodes.keys()):
        spec = opcodes[opcode]
        opselect = DeciderCombinator(left=op_signal, cmp="=", right=opcode,
                                     out=Signals.EVERY, dir=Direction.EAST,
                                     pos=(0, height))
        entities.append(opselect)
        params = spec['params']
        func_model = spec['make'](regs, operands)
        func_model.connect_input("state_in", opselect.output, 'green')

        for reg in Params.all('fast_write', params):
            out = 'reg_' + reg
            if reg not in fast_write_lines:
                rm = reg_models[reg]
                rm.connect_input("fast_write", opselect.output, 'red')
                rm.connect_input("d_write", func_model.get_output(out), 'red')
            else:
                opselect.output.join(fast_write_lines[reg], 'red')
                func_model.connect_output(out, reg_write_lines[reg], 'red')
            reg_write_lines[reg] = func_model.get_output(out)
            fast_write_lines[reg] = opselect.output

        for reg in Params.all('slow_write', params):
            assert reg in reg_write_lines # For now
            func_model.connect_output('reg_' + reg, reg_write_lines[reg], 'red')
            func_model.connect_output('reg_' + reg + '_clock',
                reg_models[reg].get_input('clock'), 'green')

        if Params.WRITE_PC in params:
            func_model.connect_output("pc", pc.get_input("pc_set"), 'red')
            func_model.connect_output("pc_clock", pc.get_input("clock"), 'green')

        if Params.DISABLE_AUTO_INC_PC in params:
            disable = make_disable_inc_pc(opcode, Signals.P, op_signal)
            disable.connect_input("insn", opselect.input, 'red')
            disable.connect_output("pc", pc.get_input("pc_set"), 'red')
            disable.shift(3, height)
            func_model.shift(6, 0)
            entities.extend(disable.entities)

        if Params.RAM_READ in params:
            func_model.connect_output('ram_addr', ram.get_input('read_addr'),
                                      'green')
            func_model.connect_input('ram_res', ram.get_output('data'), 'red')

        if Params.RAM_WRITE in params:
            func_model.connect_output('ram_data', ram.get_input('write'),
                                      'green')
            func_model.connect_input('ram_write_fin',
                                     ram.get_output('write_fin'), 'green')

        if prev_op_sel is None:
            head = opselect
        else:
            prev_op_sel.input.join(opselect.input, 'red')
            prev_op_sel.input.join(opselect.input, 'green')

        func_model.shift(3, height)
        entities.extend(func_model.entities)
        prev_op_sel = opselect
        height += 1
    model = Model(*entities)
    model.define_input("op_in", head.input, {'red'})
    model.define_input("reg_in", head.input, {'green'})
    return model

def build_cpu():
    bp = Blueprint()

    rom = make_rom_select(8, Signals.P)
    regA = make_register(Signals.A)
    regA.shift(0, -2)
    regB = make_register(Signals.B)
    regB.shift(0, -3)
    # Connect output data together
    regA.connect_output("data", regB.get_output("data"), 'green')

    pc = make_program_counter(Signals.P)
    pc.shift(5, 0)

    ram = make_ram(8)
    ram.shift(8, -3) # temp location. was 12, -4

    model = build_opcodes({
        'A': regA,
        'B': regB
    }, pc, ram, {
        'A': Signals.A,
        'B': Signals.B
    }, [
        Signals.N1,
        Signals.N2
    ])
    regA.connect_output("data", model.get_input("reg_in"), 'green')

    combined = Model.compose(regA, regB, model)
    combined.define_input("op_in", model.get_input("op_in"), {'red'})

    rom.connect_output("insn", combined.get_input("op_in"), 'red')

    rom.shift(-5, 0)

    op_forward = DeciderCombinator(left=Signals.O, cmp="≠", right=0,
                                   out=Signals.EVERY, dir=Direction.EAST,
                                   pos=(1, -1))
    op_forward.input.join(combined.get_input("op_in"), 'red')
    op_forward.output.join(pc.get_input("fast_inc"), 'red')
    bp.add(op_forward)

    cycle_trigger = DeciderCombinator(left=Signals.T, cmp="=", right=1,
                                      out=Signals.P, dir=Direction.WEST,
                                      pos=(-3.5, -1))
    cycle_trigger.output.join(rom.get_input("pc"), 'red')

    bridge_pole = ElectricPole(pos=(3, -1))
    bp.add(bridge_pole)
    
    cycle_trigger.input.join(bridge_pole.connections.first, 'red')
    pc.connect_output("pc", bridge_pole.connections.first, 'red')
    bp.add(cycle_trigger)

    trigger_activator = DeciderCombinator(left=Signals.C, cmp="≠", right=0,
                                          out=Signals.T, cp_count=False,
                                          dir=Direction.WEST, pos=(5, -1))
    trigger_activator.input.join(pc.get_input("clock"), 'green')
    trigger_activator.output.join(bridge_pole.connections.first, 'red')
    bp.add(trigger_activator)

    rom.add_to_blueprint(bp)
    combined.add_to_blueprint(bp)
    pc.add_to_blueprint(bp)
    ram.add_to_blueprint(bp)

    return bp

if __name__ == '__main__':
    bp = build_cpu()
    print(bp.as_exchange_str())
