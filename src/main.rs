use std::fs;
use text_io::read;
use std::time::Instant;
use rand::prelude::*;
use rand::rngs::SmallRng;
use ordered_float::OrderedFloat;

use crate::sokoengine::{Stringable, SokoInterface, MapTile, Entity};
use crate::kartal_baseline::{GenState, eval_state};

mod sokoengine;
mod sokoset;
mod mcts;
mod heuristics;
mod kartal_baseline;

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

fn game_loop_basic() -> () {
    let manager = sokoengine::SokoManager::new(sokoengine::mk_type_to_text);
    //let contents = fs::read_to_string("./src/levels/sokoban_1_t.lvl").expect("Couldn't read the file!");
    let contents = fs::read_to_string("./src/levels/sokoban_mid.lvl").expect("Couldn't read the file!");
    // Create the memory object that holds all the states
    let mut s_mem: sokoengine::SokoMemory<sokoengine::MapTile, sokoengine::Entity, sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>>
        = sokoengine::SokoMemory::from_str(&contents, &manager);
    let helper = heuristics::HeuristicHelper::new(&s_mem.current_state, &manager);
    println!("LEVEL: \n{}", s_mem.to_str(&manager));
    loop {
        // Print the heuristic value
        let h = heuristics::matching_heuristic(&s_mem.current_state, &helper);
        let h2 = heuristics::matching_heuristic_inv(&s_mem.current_state, &helper);
        let h3 = heuristics::matching_heuristic_with_extras(&s_mem.current_state, &helper);
        println!("Heuristic: {} -> {}, {}", h, h2, h3);
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

fn game_loop_set() -> () {
    let submanager: sokoengine::SokoManager<sokoset::MapSet, sokoset::EntitySet> =
        sokoengine::SokoManager::new(sokoset::mk_set_to_text);
    let patterns = sokoset::basic_patterns();
    let manager = sokoset::SetManager::new(submanager, patterns);
    //let contents = fs::read_to_string("./src/levels/sokoban_1_t.lvl").expect("Couldn't read the file!");
    let contents = fs::read_to_string("./src/levels/sokoban_multiple_players.lvl").expect("Couldn't read the file!");
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

fn searching() {
    let manager: sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>
        = sokoengine::SokoManager::new(sokoengine::mk_type_to_text);
    /*
    let submanager: sokoengine::SokoManager<sokoset::MapSet, sokoset::EntitySet> =
        sokoengine::SokoManager::new(sokoset::mk_set_to_text);
    let patterns = sokoset::Patterns::new();
    let manager = sokoset::SetManager::new(submanager, patterns);
    */
    //let contents = fs::read_to_string("./src/levels/sokoban_1_t.lvl").expect("Couldn't read the file!");
    let contents = fs::read_to_string("./src/levels/sokoban_mid.lvl").expect("Couldn't read the file!");
    //let contents = fs::read_to_string("./src/levels/sokoban_simple.lvl").expect("Couldn't read the file!");
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
    let s_init_tagged = (s_init.clone(), 0);
    /*
    let settings = mcts::SearchSettings::new(
        // Exploration Bonus
        OrderedFloat(2.0),
        // Exploitation Scale
        OrderedFloat(10.0),
        // Maximization Bias
        OrderedFloat(0.5),
        // Epsilon
        OrderedFloat(0.65),
        // Selection Policy
        mcts::SelectPolicy::EpsilonGreedy
    );
    */
    // Settings for the new heuristic (which is smaller than the old one)
    let settings = mcts::SearchSettings::new(
        // Exploration Bonus
        OrderedFloat(2.0),
        // Exploitation Scale
        // OLD: 25.0
        // OLD2: 20.0
        OrderedFloat(20.0),
        // Maximization Bias
        OrderedFloat(0.5),
        // Epsilon
        OrderedFloat(0.65),
        // Inherent Value
        // OLD: 0.0
        // OLD2: 10.0 // 81s for e-greedy
        OrderedFloat(10.0),
        // Selection Policy
        mcts::SelectPolicy::Softmax,
        //mcts::SelectPolicy::EpsilonGreedy,
        // Rollout length
        Some(25),
        // N Rollouts per leaf
        1,
        // Gamma
        OrderedFloat(0.97),
    );
    let mut s_tree = mcts::SearchTree::new(s_init_tagged, settings);
    /*
    let win = s_tree.mcts(Some(50),
        |s| { heuristics::matching_heuristic_inv(&s.0, &helper) },
        Some(1000),
        &manager,
        &mut rng);
    */
    //let heuristic = |s: &mcts::TaggedSokoState| -> OrderedFloat<f64> { heuristics::matching_heuristic_inv(&s.0, &helper) };
    let heuristic = |s: &mcts::TaggedSokoState<MapTile, Entity>| -> OrderedFloat<f64> { heuristics::matching_heuristic_with_extras(&s.0, &helper) };

    let now = Instant::now();
    /*
    let win = s_tree.mcts(heuristic,
        Some(400000),
        //Some(5),
        &manager,
        &mut rng);
    */
    let win = s_tree.mcts_episodic(heuristic,
        // Max states
        Some(400000),
        //Some(100000),
        // Max iters (# rollouts per episode)
        Some(500),
        // Min info (# samples required to stop an episode)
        //Some(700),
        None,
        &manager,
        &mut rng);
    let _ = s_tree.write_dag("output/tree_nodes.csv", "output/tree_edges.csv");
    println!("{:?}", now.elapsed());
    //let rolled_out = s_tree.rollout(&s_init, Some(100), &mut rng, &manager);
    //println!("{}", rolled_out.to_str(&manager));
    match win {
        Some(w) => { println!("WON:\n{}", w.to_str(&manager));
            println!("After exploring {} states,", s_tree.t);
            let states = s_tree.unroll_search(w.clone(), s_tree.initial_state.clone()).expect("No path found!");
            println!("Found a solution of length {}!", states.len());
            for state in states.into_iter() {
                println!("{}", state.to_str(&manager));
            }
        },
        None => {
            let best = s_tree.best_so_far();
            match best {
                Some(b) => { println!("BEST:\n{}", b.to_str(&manager)) }
                None => {}
            }
            let best_h = s_tree.best_heuristic_so_far(heuristic);
            match best_h {
                Some(b) => { println!("BEST_H:\n{}", b.to_str(&manager)) }
                None => {}
            }
        }
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


fn kartal() -> () {
    let manager: sokoengine::SokoManager<sokoengine::MapTile, sokoengine::Entity>
        = sokoengine::SokoManager::new(sokoengine::mk_type_to_text);
    let mut rng = SmallRng::seed_from_u64(SEED_VALUE);
    let settings = mcts::SearchSettings::new(
        // Exploration Bonus
        OrderedFloat(2.0),
        // Exploitation Scale
        OrderedFloat(20.0),
        // Maximization Bias
        OrderedFloat(0.0),
        // Epsilon
        OrderedFloat(0.65),
        // Inherent Value (does not really exist in this setting)
        OrderedFloat(0.0),
        // Selection Policy
        mcts::SelectPolicy::Softmax,
        //mcts::SelectPolicy::EpsilonGreedy,
        // Rollout length
        None,
        // N Rollouts per leaf
        1,
        // Gamma
        OrderedFloat(0.97),
    );
    let s_init = GenState::new(5,5);
    let now = Instant::now();
    let heuristic = |s: &GenState| -> OrderedFloat<f64> { eval_state(s) };
    let mut s_tree = mcts::SearchTree::new(s_init, settings);
    let _ = s_tree.mcts(heuristic,
        None,
        Some(20000),
        None,
        &manager,
        &mut rng);
    println!("{:?}", now.elapsed());
    let best = s_tree.best_so_far();
    match best {
        Some(b) => { println!("BEST:\n{}", b.level_init.to_str(&manager)) }
        None => {}
    }
    let best_h = s_tree.best_heuristic_so_far(heuristic);
    match best_h {
        Some(b) => { println!("BEST_H:\n{}", b.level_init.to_str(&manager)) }
        None => {}
    }
}

fn main() {
    //game_loop_set();
    //game_loop_basic();
    //searching();
    kartal();
}

