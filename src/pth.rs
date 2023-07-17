use color_eyre::{
    eyre::{bail, eyre},
    Result,
};
use log::debug;
use std::collections::HashMap;
use std::io::Read;
use std::path::PathBuf;

use crate::pickle::PyValue;
use crate::pickle::{self};

fn first_persid(pyv: &PyValue) -> Option<PyValue> {
    let mut persid = None;
    pyv.visit(&mut |pv: &PyValue| {
        if let PyValue::PersId(inner) = pv {
            persid = Some(*inner.clone());
            false
        } else {
            true
        }
    });
    persid
}

#[derive(Debug, Clone, Copy)]
enum PthDtype {
    FP32,
    FP16,
    BF16,
}

impl PthDtype {
    fn size(self) -> usize {
        match self {
            PthDtype::FP32 => 4,
            PthDtype::FP16 | PthDtype::BF16 => 2,
        }
    }
}

impl TryFrom<&str> for PthDtype {
    type Error = color_eyre::eyre::Error;

    fn try_from(value: &str) -> std::result::Result<Self, Self::Error> {
        use PthDtype::*;
        Ok(match value {
            "Float32Storage" => FP32,
            "Float16Storage" => FP16,
            "BFloat16Storage" => BF16,
            _ => bail!("unknown torch dtype"),
        })
    }
}

#[derive(Debug, Clone)]
pub struct PthTensorLoc {
    dtype: PthDtype,
    zipfile: PathBuf,
    zip_inner_file: String,
    nelements: usize,
    shape: Vec<usize>,
}

impl safetensors::View for PthTensorLoc {
    fn dtype(&self) -> safetensors::Dtype {
        match self.dtype {
            PthDtype::FP32 => safetensors::Dtype::F32,
            PthDtype::FP16 => safetensors::Dtype::F16,
            PthDtype::BF16 => safetensors::Dtype::BF16,
        }
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn data(&self) -> std::borrow::Cow<[u8]> {
        debug!("loading pth tensor {:?}", self);
        let zf = std::fs::File::open(&self.zipfile).expect("open pth zip");
        let mut reader = std::io::BufReader::new(zf);
        let mut za = zip::ZipArchive::new(&mut reader).expect("ziparchive");
        let mut df = za
            .by_name(self.zip_inner_file.as_str())
            .expect("zip inner file");
        let mut bytes = vec![0u8; self.data_len()];
        df.read_exact(&mut bytes[..]).expect("read pth tensor");
        bytes.into()
    }

    fn data_len(&self) -> usize {
        self.nelements * self.dtype.size()
    }
}

#[derive(Debug)]
pub struct PthReader {
    pub(crate) tensors: HashMap<String, PthTensorLoc>,
}

impl PthReader {
    pub fn new(paths: Vec<PathBuf>) -> Result<Self> {
        let mut pickles = vec![];
        let mut tensors: HashMap<String, PthTensorLoc> = Default::default();
        for path in paths.iter() {
            debug!("read zip: {:?}", path);
            let file = std::fs::File::open(path)?;
            let mut reader = std::io::BufReader::new(file);
            let mut za = zip::ZipArchive::new(&mut reader)?;
            let mut pklname = None;
            for name in za.file_names() {
                if name.ends_with("data.pkl") {
                    pklname = Some(name.to_owned());
                }
            }
            debug!("pkl: {:?}", pklname);
            if let Some(pklname) = pklname {
                let zf = za
                    .by_name(&pklname)
                    .map_err(|e| eyre!("zip error: {:?}", e))?;
                let mut pklreader = std::io::BufReader::new(zf);
                let pv = pickle::unpickle(&mut pklreader)?;
                let mut srd = None;
                pv.visit(&mut |pyv: &PyValue| {
                    if let PyValue::Dict(kvs) = pyv {
                        if let Some((PyValue::String(_s), PyValue::Reduce(rb))) = kvs.first() {
                            if let PyValue::Global(m, n) = &rb.0 {
                                debug!("g: {m} {n}");
                                if m == "torch._utils" && n == "_rebuild_tensor_v2" {
                                    srd = Some(pyv.clone());
                                    return false;
                                }
                            }
                        }
                    };
                    true
                });
                if let Some(PyValue::Dict(kvs)) = srd {
                    for (k, v) in kvs {
                        if let (PyValue::String(tensorname), PyValue::Reduce(rb)) = (k, v) {
                            if let PyValue::Global(m, n) = rb.0 {
                                if m == "torch._utils" && n == "_rebuild_tensor_v2" {
                                    if let PyValue::Tuple(ivs) = &rb.1 {
                                        let shape: Vec<usize> =
                                            if let PyValue::Tuple(ref svs) = ivs[2] {
                                                svs.iter()
                                                    .map(|si| {
                                                        if let PyValue::Int(i) = si {
                                                            *i as usize
                                                        } else {
                                                            todo!("aaagh");
                                                        }
                                                    })
                                                    .collect()
                                            } else {
                                                vec![]
                                            };
                                        let pid = first_persid(&rb.1);
                                        if let Some(PyValue::Tuple(tvs)) = pid {
                                            let (pklbase, _pkln) =
                                                pklname.rsplit_once('/').unwrap();
                                            if let (
                                                PyValue::Global(_torch, storage),
                                                PyValue::String(zfn),
                                                PyValue::Int(size),
                                            ) = (&tvs[1], &tvs[2], &tvs[4])
                                            {
                                                tensors.insert(
                                                    tensorname,
                                                    PthTensorLoc {
                                                        dtype: storage.as_str().try_into()?,
                                                        zipfile: path.clone(),
                                                        zip_inner_file: format!(
                                                            "{pklbase}/data/{zfn}"
                                                        ),
                                                        nelements: *size as usize,
                                                        shape,
                                                    },
                                                );
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
                pickles.push(pv);
            }
        }
        Ok(Self { tensors })
    }
}
