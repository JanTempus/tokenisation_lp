use std::collections::HashMap;
use std::fmt;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TokenInstance {
    pub token: String,
    pub start: usize,
    pub end: usize,
    pub lp_value: f64,
}

impl TokenInstance {
    pub fn new(token: String, start: usize, end: usize) -> Self {
        Self {
            token,
            start,
            end,
            lp_value: -1.0,
        }
    }
}


#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PossibleToken {
    pub token: String,
    pub lp_value: f64,
}

impl PossibleToken {
    pub fn new(token: String) -> Self {
        Self {
            token,
            lp_value: -1.0,
        }
    }
}

use sprs::{CsMat, TriMat};
use std::collections::HashSet;

pub fn solve_lp_vec(
    edges_list: Vec<Vec<TokenInstance>>,
    edge_list_weight: Vec<i32>,
    tokens: Vec<TokenInstance>,
    free_edges_list: Vec<Vec<TokenInstance>>,
    num_vertices_list: Vec<usize>,
) {
    let num_strings = edges_list.len();
    if num_strings != free_edges_list.len() {
        panic!("Mismatch between edges list and free edges list");
    }

    let num_tokens = tokens.len();

    println!("Started working on the LP");

    let mut a_matrices = Vec::new();
    let mut b_matrices = Vec::new();
    let mut m_matrices = Vec::new();
    let mut b_vectors = Vec::new();
    let mut free_w_vectors = Vec::new();
    let mut nonfree_w_vectors = Vec::new();

    for i in 0..num_strings {
        if i % 1000 == 0 {
            println!("{} / {}", i, num_strings);
        }

        let edges = &edges_list[i];
        let free_edges = &free_edges_list[i];
        let num_edges = edges.len();
        let num_free_edges = free_edges.len();
        let num_vertices = num_vertices_list[i];

        // A matrix
        let mut a = TriMat::new((num_vertices, num_edges));
        for (idx, edge) in edges.iter().enumerate() {
            a.add_triplet(edge.start, idx, 1.0);
            a.add_triplet(edge.end, idx, -1.0);
        }

        // B matrix
        let mut b = TriMat::new((num_vertices, num_free_edges));
        for (idx, edge) in free_edges.iter().enumerate() {
            b.add_triplet(edge.start, idx, 1.0);
            b.add_triplet(edge.end, idx, -1.0);
        }

        // M matrix
        let mut m = TriMat::new((num_edges, num_tokens));
        for (j, edge) in edges.iter().enumerate() {
            if let Some(index) = tokens.iter().position(|t| t.token == edge.token) {
                m.add_triplet(j, index, 1.0);
            }
        }

        a_matrices.push(a.to_csr());
        b_matrices.push(b.to_csr());
        m_matrices.push(m.to_csr());

        // b vector
        let mut b_vec = vec![0.0; num_vertices];
        b_vec[0] = 1.0;
        b_vec[num_vertices - 1] = -1.0;
        b_vectors.push(b_vec);

        // weights
        let w_nonfree = vec![edge_list_weight[i] as f64; num_edges];
        let w_free = vec![edge_list_weight[i] as f64; num_free_edges];
        nonfree_w_vectors.push(w_nonfree);
        free_w_vectors.push(w_free);
    }

    println!("Finished setting up LP matrices");

    // Block diagonal + vertical stacking
    let big_a = sprs::stack::block_diag(&a_matrices);
    let big_b = sprs::stack::block_diag(&b_matrices);
    let big_m = sprs::stack::vstack(&m_matrices);

    let big_b_vec: Vec<f64> = b_vectors.into_iter().flatten().collect();
    let big_free_w_vec: Vec<f64> = free_w_vectors.into_iter().flatten().collect();
    let big_nonfree_w_vec: Vec<f64> = nonfree_w_vectors.into_iter().flatten().collect();

    let tokens_cap: Vec<f64> = vec![1.0; num_tokens];

    // Matrices are now ready to be passed to an LP solver like `good_lp` or `argmin`
}



#[allow(dead_code,unused_variables,unused_mut)]
fn make_matrix(edges_list:Vec<Vec<TokenInstance>>, 
               edge_list_weight: Vec<i32>,
               tokens: Vec<PossibleToken>,
               free_edges_list: Vec<Vec<TokenInstance>>,
               num_vertices_list: Vec<usize>){


    let num_strings = edges_list.len();
    if num_strings != free_edges_list.len() {
        panic!("Mismatch between edges list and free edges list");
    }

    let num_tokens = tokens.len();



    let mut a_matrices: Vec<CsMat<isize>> = Vec::new();
    let mut b_matrices: Vec<CsMat<isize>> = Vec::new();
    let mut m_matrices: Vec<CsMat<isize>> = Vec::new();
    let mut b_vectors :Vec<Vec<isize>> = Vec::new();
    let mut free_w_vectors: Vec<Vec<isize>> = Vec::new();
    let mut nonfree_w_vectors:Vec<Vec<isize>> = Vec::new();

    for i in 0..num_strings {
        if i % 1000 == 0 {
            println!("{} / {}", i, num_strings);
        }

        let edges = &edges_list[i];
        let free_edges = &free_edges_list[i];
        let num_edges = edges.len();
        let num_free_edges = free_edges.len();
        let num_vertices = num_vertices_list[i];

        let mut a = TriMat::new((num_vertices, num_edges));
        for (idx, edge) in edges.iter().enumerate() {
            a.add_triplet(edge.start, idx, 1);
            a.add_triplet(edge.end, idx, -1);
        }
        let mut b = TriMat::new((num_vertices, num_free_edges));
        for (idx, edge) in free_edges.iter().enumerate() {
            b.add_triplet(edge.start, idx, 1);
            b.add_triplet(edge.end, idx, -1);
        }

        // M matrix
        let mut m = TriMat::new((num_edges, num_tokens));
        for (j, edge) in edges.iter().enumerate() {
            let index = tokens
            .iter()
            .position(|t| t.token == edge.token)
            .expect(&format!("Token '{}' not found in tokens list", edge.token));
        
            m.add_triplet(j, index, 1);
        }

        let a_csr=a.to_csr::<usize>();
        let b_csr=b.to_csr::<usize>();
        let m_csr=m.to_csr::<usize>();
        
        a_matrices.push(a_csr);

        // b vector
        let mut b_vec = vec![0; num_vertices];
        b_vec[0] = 1;
        b_vec[num_vertices - 1] = -1;
      

        // weights
        let w_nonfree = vec![edge_list_weight[i] as isize; num_edges];
        let w_free = vec![edge_list_weight[i] as isize; num_free_edges];
        nonfree_w_vectors.push(w_nonfree);
        free_w_vectors.push(w_free);
        
    }
}