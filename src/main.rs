use std::fs;
use ndarray::{Array, array, Ix2};
use text_io::read;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::SmallRng;

use crate::sokoengine::{Stringable, SokoInterface};
use crate::heuristics::{find_wall_traps};

mod sokoengine;
mod sokoset;
mod mcts;
mod heuristics;

const SEED_VALUE: u64 = 0;

enum Command {
    Quit,
    Undo,
    Direction { d: sokoengine::Direction },
    Nothing,
}

fn char_to_command(c: Option<&u8>) -> Command {
        match c {
            // wasd controls
            Some(119) => Command::Direction { d: sokoengine::Direction::Up },
            Some(97) => Command::Direction { d: sokoengine::Direction::Left },
            Some(115) => Command::Direction { d: sokoengine::Direction::Down },
            Some(100) => Command::Direction { d: sokoengine::Direction::Right },
            // x to quit
            Some(120) => Command::Quit,
            // u to undo
            Some(117) => Command::Undo,
            // Any other input should do nothing, but not quit
            Some(_) => Command::Nothing,
            None => Command::Nothing,
        }
}

fn game_loop() -> () {
    let submanager: sokoengine::SokoManager<sokoset::MapSet, sokoset::EntitySet> =
        sokoengine::SokoManager::new(sokoset::mk_set_to_text);
    let patterns = sokoset::Patterns::new();
    let manager = sokoset::SetManager::new(submanager, patterns);
    //let contents = fs::read_to_string("./src/sokoban_1_t.lvl").expect("Couldn't read the file!");
    let contents = fs::read_to_string("./src/sokoban_multiple_players.lvl").expect("Couldn't read the file!");
    // Create the memory object that holds all the states
    /*
    let mut s_mem: sokoengine::SokoMemory<sokoengine::MapTile, sokoengine::Entity, sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>>
        = sokoengine::SokoMemory::from_str(&contents, &manager);
    */
    let mut s_mem: sokoengine::SokoMemory<sokoset::MapSet, sokoset::EntitySet, sokoset::SetManager> =
        sokoengine::SokoMemory::from_str(&contents, &manager);
    println!("LEVEL: \n{}", s_mem.to_str(&manager));
    loop {
        // Get the text input
        let i: String = read!();
        let c: Option<&u8> = i.as_bytes().get(0);
        //println!("{:?}", c);
        let command = char_to_command(c);
        match command {
            Command::Quit => {
                    println!("YOU QUIT");
                    break;
                }
            Command::Direction { d: direction} => {
                println!("{:?}", direction);
                s_mem.update(direction, &manager);
            }
            Command::Undo => {
                s_mem.undo();
            }
            Command::Nothing => {}
        }
        //println!("{}", (*s_box).to_str(&manager));
        println!("{}", s_mem.to_str(&manager));
        if s_mem.is_win() {
            println!("YOU WIN!");
            break;
        }
    }

}


fn main() {
    let manager: sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>
        = sokoengine::SokoManager::new(sokoengine::mk_type_to_text);
    /*
    let submanager: sokoengine::SokoManager<sokoset::MapSet, sokoset::EntitySet> =
        sokoengine::SokoManager::new(sokoset::mk_set_to_text);
    let patterns = sokoset::Patterns::new();
    let manager = sokoset::SetManager::new(submanager, patterns);
    */
    //let contents = fs::read_to_string("./src/sokoban_1_t.lvl").expect("Couldn't read the file!");
    let contents = fs::read_to_string("./src/sokoban_simple.lvl").expect("Couldn't read the file!");
    let s_init = sokoengine::SokoState::from_str(&contents, &manager);
    /*
    // Heuristic testing!
    let helper = heuristics::HeuristicHelper::new(&s_init, &manager);
    let now = Instant::now();
    let h = heuristics::matching_heuristic(&s_init, &helper);
    println!("{:?}", now.elapsed());
    println!("{:?}", h);
    */
    let helper = heuristics::HeuristicHelper::new(&s_init, &manager);
    let mut rng = SmallRng::seed_from_u64(SEED_VALUE);
    let mut s_tree = mcts::SearchTree::new(s_init.clone());
    let win = s_tree.mcts(Some(50),
        |s| { heuristics::matching_heuristic_inv(s, &helper) },
        Some(1000),
        &manager,
        &mut rng);
    //let rolled_out = s_tree.rollout(&s_init, Some(100), &mut rng, &manager);
    //println!("{}", rolled_out.to_str(&manager));
    match win {
        Some(w) => { println!("{}", w.to_str(&manager)) },
        None => {}
    }
    /*
    // BFS testing!
    println!("Start search!");
    let now = Instant::now();
    let win = s_tree.basic_search(Some(sokoengine::SokoState::is_win), None, &manager).expect("Could not win!");
    let elapsed = now.elapsed();
    let v_states = s_tree.unroll_search(win, s_init.clone()).expect("Could not find the start!");
    for v_state in v_states {
        println!("{}", v_state.to_str(&manager));
        println!("");
    }
    println!("Explored {} states in {} seconds", s_tree.n_states(), elapsed.as_secs());
    */
}

/*
fn main() {
    game_loop()
}
*/
