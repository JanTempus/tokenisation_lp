use serde::{Deserialize, Serialize};

#[derive(Serialize, Deserialize, Debug)]
pub struct TokenInstance {
    token: String,
    start: usize,
    end: usize,
    lpValue: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PossibleToken {
    token: String,
    lpValue: f64,
}

use pyo3::prelude::*;
use pyo3::wrap_pyfunction;

#[pyfunction]
fn process_tokens(json_input: &str) -> PyResult<()> {
    let tokens: Vec<TokenInstance> = serde_json::from_str(json_input)
        .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
    
    for token in tokens {
        println!("{:?}", token);
    }
    Ok(())
}

#[pymodule]
fn rust_tokenizer(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(process_tokens, m)?)?;
    Ok(())
}
