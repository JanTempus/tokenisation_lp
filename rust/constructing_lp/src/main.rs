use serde::{Deserialize, Serialize};
use sprs::{TriMat,CsMat};
use std::fs::File;
use std::collections::HashMap;

#[derive(Serialize, Deserialize, Debug)]
pub struct TokenInstance {
    token: String,
    start: usize,
    end: usize,
    lp_value: f64,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct PossibleToken {
    token: String,
    lp_value: f64,
}
#[derive(Debug, Deserialize, Serialize)]
pub struct LPData {
    pub edges_list: Vec<Vec<TokenInstance>>,
    pub edge_list_weight: Vec<usize>,
    pub tokens: Vec<PossibleToken>,
    pub free_edges_list: Vec<Vec<TokenInstance>>,
    pub num_vertices_list: Vec<usize>,
}

pub fn print_edges(edges_list: &[TokenInstance]) {
    for edge in edges_list {
        println!(
            "start: {}, end: {}, token: '{}', lp_value: {}",
            edge.start, edge.end, edge.token, edge.lp_value
        );
    }
}



use rayon::prelude::*;

fn make_matrix_parallel(edges_list:&Vec<Vec<TokenInstance>>, 
                        edge_list_weight: &Vec<usize>,
                        tokens: &Vec<PossibleToken>,
                        free_edges_list: &Vec<Vec<TokenInstance>>,
                        num_vertices_list: &Vec<usize>)->(CsMat<isize>,
                                                          CsMat<isize>,
                                                          CsMat<isize>,
                                                          Vec<isize>,
                                                          Vec<usize>,
                                                          Vec<usize>) {

        let num_strings = edges_list.len();
        assert_eq!(num_strings, free_edges_list.len());
    
        let token_index_map: HashMap<String, usize> = tokens
            .iter()
            .enumerate()
            .map(|(i, t)| (t.token.clone(), i))
            .collect();
    
        // Compute offsets
        let vertex_offsets: Vec<usize> = std::iter::once(0)
            .chain(num_vertices_list.iter().scan(0, |acc, &n| {
                *acc += n;
                Some(*acc)
            }))
            .collect();
    
        let edge_offsets: Vec<usize> = std::iter::once(0)
            .chain(edges_list.iter().map(Vec::len).scan(0, |acc, n| {
                *acc += n;
                Some(*acc)
            }))
            .collect();
    
        let free_edge_offsets: Vec<usize> = std::iter::once(0)
            .chain(free_edges_list.iter().map(Vec::len).scan(0, |acc, n| {
                *acc += n;
                Some(*acc)
            }))
            .collect();
    
        let total_vertices = *vertex_offsets.last().unwrap();
        let total_edges = *edge_offsets.last().unwrap();
        let total_free_edges = *free_edge_offsets.last().unwrap();
        let num_tokens = tokens.len();
    
        // // Parallel processing
        let results: Vec<_> = (0..num_strings)
            .into_par_iter()
            .map(|i| {
                let curr_edges = &edges_list[i];
                let curr_free_edges = &free_edges_list[i];
                let num_vertices = num_vertices_list[i];
                let weight = edge_list_weight[i] as isize;
    
                let voff = vertex_offsets[i];
                let eoff = edge_offsets[i];
                let feoff = free_edge_offsets[i];
    
                let mut a_local:TriMat<isize>= TriMat::new((total_vertices, total_edges));
                let mut b_local:TriMat<isize> = TriMat::new((total_vertices, total_free_edges));
                let mut m_local:TriMat<isize> = TriMat::new((total_edges, num_tokens));
                let mut b_vec_local = vec![0isize; total_vertices];
    
                // A
                for (j, edge) in curr_edges.iter().enumerate() {
                    a_local.add_triplet(edge.start + voff, j + eoff, 1);
                    a_local.add_triplet(edge.end + voff, j + eoff, -1);
                }
    
                // B
                for (j, edge) in curr_free_edges.iter().enumerate() {
                    b_local.add_triplet(edge.start + voff, j + feoff, 1);
                    b_local.add_triplet(edge.end + voff, j + feoff, -1);
                }
    
                // M
                for (j, edge) in curr_edges.iter().enumerate() {
                    let idx = token_index_map
                        .get(&edge.token)
                        .expect("Token not found in token list");
                    m_local.add_triplet(j + eoff, *idx, 1);
                }
    
                // b vector
                b_vec_local[voff] = 1;
                b_vec_local[voff + num_vertices - 1] = -1;
    
                let free_w = vec![weight; curr_free_edges.len()];
                let nonfree_w = vec![weight; curr_edges.len()];
    
                (a_local, b_local, m_local, b_vec_local, nonfree_w, free_w)
            })
            .collect();
    
        // // Reduce: combine triplets and vectors
        let mut a_comb:TriMat<isize> = TriMat::new((total_vertices, total_edges));
        let mut b_comb:TriMat<isize> = TriMat::new((total_vertices, total_free_edges));
        let mut m_comb:TriMat<isize> = TriMat::new((total_edges, num_tokens));
        let mut b_vec = vec![0 ; total_vertices];
        let mut nonfree_w : Vec<usize>= Vec::with_capacity(total_edges);
        let mut free_w : Vec<usize>= Vec::with_capacity(total_free_edges);
        
        for (a, b, m, bv, nw, fw) in results {
            // Merge A
            for (val, (i,j)) in a.triplet_iter() {
                a_comb.add_triplet(i, j, *val);
            }
        
            //Merge B
            for (val, (i,j)) in b.triplet_iter() {
                b_comb.add_triplet(i, j, *val);
            }
        
            // Merge M
            for (val, (i,j)) in m.triplet_iter() {
                m_comb.add_triplet(i, j, *val);
            }
        
            // Merge b_vec (element-wise addition)
            for (i, val) in bv.into_iter().enumerate() {
                b_vec[i] += val;
            }
        
            // Merge weights
            nonfree_w.extend(nw.iter().map(|&x| x as usize));
            free_w.extend(fw.iter().map(|&x| x as usize));
        }
    
       return (
            a_comb.to_csr(),
            b_comb.to_csr(),
            m_comb.to_csr(),
            b_vec,
            nonfree_w,
            free_w,
        )
}
                        
                        

fn make_matrix(edges_list:&Vec<Vec<TokenInstance>>, 
               edge_list_weight: &Vec<usize>,
               tokens: &Vec<PossibleToken>,
               free_edges_list: &Vec<Vec<TokenInstance>>,
               num_vertices_list: &Vec<usize>)->(CsMat<isize>,
                                                CsMat<isize>,
                                                CsMat<isize>,
                                                Vec<isize>,
                                                Vec<isize>,
                                                Vec<isize>) {


    let num_strings = edges_list.len();
    if num_strings != free_edges_list.len() {
        panic!("Mismatch between edges list and free edges list");
    }


    let num_vertices:  usize = num_vertices_list.iter().sum();
    let num_edges:     usize = edges_list.iter().map(|edges| edges.len()).sum();
    let num_edges_free:usize = free_edges_list.iter().map(|edges| edges.len()).sum();

    let num_tokens = tokens.len();


    let mut free_w_vectors:   Vec<isize> = Vec::new();
    let mut nonfree_w_vectors:Vec<isize> = Vec::new();


    let mut offset_edges:      usize = 0;
    let mut offset_edges_free: usize = 0;
    let mut offset_vertices:   usize = 0;

    let mut a:TriMat<isize> = TriMat::new((num_vertices, num_edges));
    let mut b:TriMat<isize> = TriMat::new((num_vertices, num_edges_free));
    let mut m:TriMat<isize> = TriMat::new((num_edges, num_tokens));

    let mut b_vec =vec![0; num_vertices];

    for i in 0..num_strings {

        let curr_edges = &edges_list[i];
        let curr_free_edges = &free_edges_list[i];
        let curr_num_edges = curr_edges.len();
        let curr_num_free_edges = curr_free_edges.len();
        let curr_num_vertices = num_vertices_list[i];


        for (idx, edge) in curr_edges.iter().enumerate() {
            a.add_triplet(edge.start+offset_vertices, idx+offset_edges, 1);
            a.add_triplet(edge.end+offset_vertices, idx+offset_edges ,-1);
        }
        
        for (idx, edge) in curr_free_edges.iter().enumerate() {
            b.add_triplet(edge.start+offset_vertices, idx+offset_edges_free, 1);
            b.add_triplet(edge.end+offset_vertices  , idx+offset_edges_free, -1);
        }

    
        for (j, edge) in curr_edges.iter().enumerate() {
            let index = tokens
            .iter()
            .position(|t| t.token == edge.token)
            .expect(&format!("Token '{}' not found in tokens list", edge.token));
        
            m.add_triplet(j+offset_edges, index, 1);
        }


        b_vec[offset_vertices]=1;
        b_vec[offset_vertices+curr_num_vertices-1]=-1;
        offset_edges_free += curr_num_free_edges;
        offset_edges      += curr_num_edges;
        offset_vertices   += curr_num_vertices;
      

        // // weights
        let w_nonfree = vec![edge_list_weight[i] as isize; curr_num_edges];
        let w_free = vec![edge_list_weight[i] as isize; curr_num_free_edges];
        nonfree_w_vectors.extend(w_nonfree);
        free_w_vectors.extend(w_free);
        
    }

    let a_csr=a.to_csr::<usize>();
    let b_csr=b.to_csr::<usize>();
    let m_csr=m.to_csr::<usize>();

    return (a_csr,b_csr,m_csr,b_vec,nonfree_w_vectors,free_w_vectors)

    
}   


pub fn load_lp_data(file_path: &str) -> LPData {
    let file = File::open(file_path).expect("file not found");
    let lp_data: LPData = serde_json::from_reader(file).expect("error while reading");
    return lp_data
}

fn csr_components(mat: &CsMat<isize>)
    -> (Vec<isize>, Vec<usize>, Vec<usize>, usize, usize)
{
    let data    = mat.data().to_vec();
    let indices = mat.indices().to_vec();
    let indptr = mat.indptr().as_slice().unwrap().to_vec();
    (data, indices, indptr, mat.rows(), mat.cols())
}


#[allow(dead_code)]
fn main() {
    let lp_data = load_lp_data("/home/jantempus/Desktop/Projects/NLP/tokenisation_lp/python/lp_data.json");
    
    let (_a_mat,_b_mat, _m_mat,_b_vec,_w_vec,_v_vec)=make_matrix(&lp_data.edges_list, 
                                                                 &lp_data.edge_list_weight,
                                                                 &lp_data.tokens,
                                                                 &lp_data.free_edges_list,
                                                                 &lp_data.num_vertices_list );

    let (_a_mat1,_b_mat, _m_mat,_b_vec,_w_vec,_v_vec)=make_matrix_parallel(&lp_data.edges_list, 
                                                                 &lp_data.edge_list_weight,
                                                                 &lp_data.tokens,
                                                                 &lp_data.free_edges_list,
                                                                 &lp_data.num_vertices_list );

            
    println!("{:#?}",  _a_mat);
    println!("{:#?}",  _a_mat1);
}
