use std::path::PathBuf;

use color_eyre::Result;
use structopt::StructOpt;

mod pickle;
mod pth;


#[derive(StructOpt, Debug)]
struct Opt {
    #[structopt(parse(from_os_str))]
    input_files: Vec<PathBuf>,
    #[structopt(parse(from_os_str))]
    output_file: PathBuf,
}

fn main() -> Result<()> {
    pretty_env_logger::init();
    color_eyre::install()?;
    let opt = Opt::from_args();
    let pthreader = pth::PthReader::new(opt.input_files)?;
    safetensors::serialize_to_file(pthreader.tensors, &None, &opt.output_file)?;
    Ok(())
}
