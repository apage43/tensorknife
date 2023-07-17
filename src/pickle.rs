use std::collections::HashMap;

use byteorder::{LittleEndian, ReadBytesExt};
use color_eyre::{eyre::bail, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PyValue {
    Mark,
    Bool(bool),
    Tuple(Vec<PyValue>),
    Int(i64),
    String(String),
    Global(String, String),
    Bytes(Vec<u8>),
    Dict(Vec<(PyValue, PyValue)>),
    Reduce(Box<(PyValue, PyValue)>),
    PersId(Box<PyValue>)
}

pub trait PyValueVisitor {
    fn visit(&mut self, item: &PyValue) -> bool;
}

impl PyValue {
    pub fn visit(&self, visitor: &mut impl PyValueVisitor) {
        if visitor.visit(self) {
            match self {
                PyValue::Tuple(ivs) => {
                    for v in ivs {
                        v.visit(visitor);
                    }
                },
                PyValue::Dict(dvs) => {
                    for (k,v) in dvs {
                        k.visit(visitor);
                        v.visit(visitor);
                    }
                },
                PyValue::Reduce(rb) => {
                    rb.0.visit(visitor);
                    rb.1.visit(visitor);
                },
                PyValue::PersId(pid) => {
                    pid.visit(visitor);
                },
                _ => {}
            }
        }
    }
}

impl <F> PyValueVisitor for F where F: FnMut(&PyValue) -> bool {
    fn visit(&mut self, item: &PyValue) -> bool {
        self(item)
    }
}

#[derive(Default)]
pub struct Unpickler {}
impl Unpickler {
    pub fn load(&mut self, input: &mut impl std::io::BufRead) -> Result<PyValue> {
        let mut stack: Vec<PyValue> = vec![];
        let mut memo: HashMap<usize, PyValue> = Default::default();
        loop {
            let key = input.read_u8()?;
            match key {
                0x80 => {
                    let proto = input.read_u8()?;
                    //eprintln!("pickle protocol: {proto}");
                    if proto > 4 {
                        bail!("unsupported pickle protocol");
                    }
                }
                b'q' => {
                    let binp = input.read_u8()?;
                    //eprintln!("BINPUT {binp}");
                    if let Some(val) = stack.last() {
                        memo.insert(binp as usize, val.clone());
                    } else {
                        bail!("binput with nothing on stack");
                    }
                }
                b'r' => {
                    let binp = input.read_u32::<LittleEndian>()?;
                    //eprintln!("LONG_BINPUT {binp}");
                    if let Some(val) = stack.last() {
                        memo.insert(binp as usize, val.clone());
                    } else {
                        bail!("binput with nothing on stack");
                    }
                }
                b'h' => {
                    let binp1 = input.read_u8()?;
                    stack.push(memo.get(&(binp1 as usize)).unwrap().clone());
                    //eprintln!("BINGET {binp1}");
                }
                b'J' => {
                    let binint = input.read_i32::<LittleEndian>()?;
                    stack.push(PyValue::Int(binint.into()));
                    //eprintln!("BININT {binint}");
                }
                b'K' => {
                    let binint = input.read_u8()?;
                    stack.push(PyValue::Int(binint.into()));
                    //eprintln!("BININT1 {binint}");
                }
                b'M' => {
                    let binint = input.read_u16::<LittleEndian>()?;
                    stack.push(PyValue::Int(binint.into()));
                    //eprintln!("BININT2 {binint}");
                }
                b'u' => {
                    //eprintln!("SETITEMS");
                    let mut items: Vec<PyValue> = vec![];
                    while let Some(item) = stack.pop() {
                        if item == PyValue::Mark {
                            break;
                        }
                        items.push(item);
                    }
                    items.reverse();
                    let mut dict = stack.pop().unwrap();
                    match dict {
                        PyValue::Dict(ref mut dictitems) => {
                            for kv in items.chunks_exact(2) {
                                dictitems.push((kv[0].clone(), kv[1].clone()))
                            }
                        },
                        _ => bail!("setitems on not a dict")
                    };
                    stack.push(dict);
                    
                }
                b'R' => {
                    //eprintln!("REDUCE");
                    let args = stack.pop().unwrap();
                    let callable = stack.pop().unwrap();
                    stack.push(PyValue::Reduce(Box::new((callable, args))));
                }
                b'}' => {
                    stack.push(PyValue::Dict(vec![]));
                    //eprintln!("EMPTY_DICT");
                }
                b'(' => {
                    //eprintln!("MARK");
                    stack.push(PyValue::Mark);
                }
                b')' => {
                    //eprintln!("EMPTY_TUPLE");
                    stack.push(PyValue::Tuple(vec![]));
                }
                b't' => {
                    // eprintln!("TUPLE");
                    let mut items: Vec<PyValue> = vec![];
                    while let Some(item) = stack.pop() {
                        if item == PyValue::Mark {
                            break;
                        }
                        items.push(item);
                    }
                    items.reverse();
                    stack.push(PyValue::Tuple(items));
                }
                0x85 => {
                    // eprintln!("TUPLE1");
                    let i1 = if let Some(i) = stack.pop() {
                        i
                    } else {
                        bail!("nothing to pop");
                    };
                    stack.push(PyValue::Tuple(vec![i1]))
                }
                0x86 => {
                    // eprintln!("TUPLE2");
                    let i1 = if let Some(i) = stack.pop() {
                        i
                    } else {
                        bail!("nothing to pop");
                    };
                    let i2 = if let Some(i) = stack.pop() {
                        i
                    } else {
                        bail!("nothing to pop");
                    };
                    stack.push(PyValue::Tuple(vec![i2, i1]))
                }
                0x87 => {
                    // eprintln!("TUPLE3");
                    let i1 = if let Some(i) = stack.pop() {
                        i
                    } else {
                        bail!("nothing to pop");
                    };
                    let i2 = if let Some(i) = stack.pop() {
                        i
                    } else {
                        bail!("nothing to pop");
                    };
                    let i3 = if let Some(i) = stack.pop() {
                        i
                    } else {
                        bail!("nothing to pop");
                    };
                    stack.push(PyValue::Tuple(vec![i3, i2, i1]))
                }
                0x88 => {
                    // eprintln!("NEWTRUE");
                    stack.push(PyValue::Bool(true));
                }
                0x89 => {
                    // eprintln!("NEWFALSE");
                    stack.push(PyValue::Bool(false));
                }
                b'X' => {
                    let rlen = input.read_u32::<LittleEndian>()?;
                    let mut strin = vec![0u8; rlen as usize];
                    input.read_exact(&mut strin)?;
                    let decoded = String::from_utf8_lossy(&strin);
                    // eprintln!("BINUNICODE {decoded:?}");
                    stack.push(PyValue::String(decoded.to_string()));
                }
                b'c' => {
                    let mut module = String::new();
                    input.read_line(&mut module)?;
                    let mut name = String::new();
                    input.read_line(&mut name)?;
                    // eprintln!("GLOBAL {:?} {:?}", module.trim_end(), name.trim_end());
                    stack.push(PyValue::Global(module.trim_end().to_string(), name.trim_end().to_string()));
                }
                b'Q' => {
                    // eprintln!("BINPERSID");
                    let id = stack.pop().expect("persid");
                    stack.push(PyValue::PersId(Box::new(id)));
                }
                b'.' => {
                    // eprintln!("STOP");
                    break;
                }
                _ => {
                    bail!(
                        "unpickler: unknown op 0x{key:02x} {:?}",
                        char::from_u32(key.into())
                    )
                }
            }
        }
        if let Some(result) = stack.pop() {
            if !stack.is_empty() {
                bail!("extra values left on stack");
            }
            Ok(result)
        } else {
            bail!("nothing left on stack");
        }
    }
}
