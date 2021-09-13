import {Canvas} from '../../animlib/src/classes/Canvas.js'
import {Image} from '../../animlib/src/classes/Image.js'
import {Path} from '../../animlib/src/classes/Path.js'
import {Rectangle} from '../../animlib/src/classes/Rectangle.js'
import {TextObject} from '../../animlib/src/classes/TextObject.js'
import {NQColors, NQTextColors, NQFonts} from '../../animlib/src/variables/NQ_colors_and_fonts.js'

/************************************************
 Parameters
************************************************/
let canvas_width = 1930
let canvas_height = 1090
let div_step_x = canvas_width * (1/24)
let div_step_y = canvas_height * (1/24)
let hook_base_pos_x = 3*div_step_x
let hook_base_pos_y = 1000
let rect_start_x = 2*div_step_x + 0.2*div_step_x 
let rect_start_y = hook_base_pos_y - 1.6*div_step_y
let section1_pos_x = 4*div_step_x
let rect_bigger_height = 200
let rect_smaller_height = 66
let rect_width = 300
let section2_pos_x = 9*div_step_x + 0.5*div_step_x
let section3_pos_x = 15*div_step_x - 0.5*div_step_x
let textf_width = 7*div_step_x - 0.5*div_step_x
let textf_height = 8*div_step_y
let textf_pos_x = div_step_x * 15
let textf_pos_y = div_step_y * 20
let bottom_pos = 3*div_step_y
let tf_exp_1_str = "Anaconda installer sets up <i>conda</i>, the package manager of Anaconda."
let tf_exp_2_str = "<i>conda</i> creates the so-called <i>base environment</i> and installs Python executable."
let tf_exp_3_str = "You can install additional libraries and packages when needed."
let tf_exp_4_str = "You can create additional environments alongside the base environment. Additional environments can contain varying sets of packages."
let tf_exp_5_str = "You can also create an environment with R for standalone usage or for combined usage with Python!"

export {
  canvas_width,
  canvas_height,
  hook_base_pos_x,
  hook_base_pos_y,
  div_step_x,
  rect_start_x,
  rect_start_y,
  section1_pos_x,
  bottom_pos,
  rect_bigger_height,
  rect_width,
  rect_smaller_height,
  section2_pos_x,
  section3_pos_x,
  textf_width,
  textf_height,
  textf_pos_x,
  textf_pos_y,
  tf_exp_1_str,
  tf_exp_2_str,
  tf_exp_3_str,
  tf_exp_4_str,
  tf_exp_5_str,
}

/************************************************
 canvas
************************************************/

// Scalable canvas
let canvas = new Canvas({})
canvas.svg.attr('viewBox','0 0 ' + canvas_width + ' ' + canvas_height)
    .attr('preserveAspectRatio','xMidYMid meet')
    .style("width", '98%')
    .style("height", '98%')
    .style("display", "block")
    .style("position", "absolute")
export {canvas}

/************************************************
 Init hook and its track
************************************************/

let params_img_hook  = {
    pos: [hook_base_pos_x, hook_base_pos_y],
    path: "../pics/hook.svg",
    id: "img_hook",
    relSize: 10,
  }
export let img_hook = new Image(params_img_hook, canvas)

let params_line_hook = {
    id: "line_hook",
    data: [[3*div_step_x, hook_base_pos_y], [3*div_step_x+3*6*div_step_x, hook_base_pos_y]],
    strokeColor: NQColors.dark_grey,
    strokeWidth: 4,
}
export let line_hook = new Path(params_line_hook, canvas)

/************************************************
 Init environment section separator lines
************************************************/

let params_line_sep1 = {
  id: "line_sep1",
  data: [[3*div_step_x+1*5*div_step_x, bottom_pos-5], [3*div_step_x+1*5*div_step_x, 700]],
  strokeColor: NQColors.steelblue,
  strokeWidth: 4,
}
export let line_sep1 = new Path(params_line_sep1, canvas)

let params_line_sep2 = {
  id: "line_sep2",
  data: [[3*div_step_x+2*5*div_step_x, bottom_pos-5], [3*div_step_x+2*5*div_step_x, 700]],
  strokeColor: NQColors.steelblue,
  strokeWidth: 4,
}
export let line_sep2 = new Path(params_line_sep2, canvas)

/************************************************
 Init blocks representing programs/packages in base environment
************************************************/

// base environment Python program
let params_rect_py1 = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_py1",
  "dim"         : [rect_width, rect_bigger_height],
  "color"       : "#006600",
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_py1 = new Rectangle(params_rect_py1, canvas)

// base environment Python packages
let params_rect_pyp1 = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_pyp1",
  "dim"         : [rect_width, rect_smaller_height],
  "color"       : NQColors.light_purple,
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_pyp1 = new Rectangle(params_rect_pyp1, canvas)

let params_rect_pyp2 = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_pyp2",
  "dim"         : [rect_width, rect_smaller_height],
  "color"       : NQColors.orange,
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_pyp2 = new Rectangle(params_rect_pyp2, canvas)

/************************************************
 Init blocks representing programs/packages in additional environment
************************************************/

// base environment Python program
let params_rect_py2 = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_py2",
  "dim"         : [rect_width, rect_bigger_height],
  "color"       : "#00b300",
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_py2 = new Rectangle(params_rect_py2, canvas)

// base environment Python packages
let params_rect_pyp3 = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_pyp3",
  "dim"         : [rect_width, rect_smaller_height],
  "color"       : NQColors.yellow,
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_pyp3 = new Rectangle(params_rect_pyp3, canvas)

/************************************************
 Init blocks representing programs/packages in Python + R environment
************************************************/

// Python program
let params_rect_py3 = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_py3",
  "dim"         : [rect_width, rect_bigger_height],
  "color"       : "#006600",
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_py3 = new Rectangle(params_rect_py3, canvas)

// R program
let params_rect_r = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_r",
  "dim"         : [rect_width, rect_bigger_height],
  "color"       : NQColors.steelblue,
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_r = new Rectangle(params_rect_r, canvas)

// base environment Python packages
let params_rect_rp = {
  "pos"         : [rect_start_x, rect_start_y],
  "id"          : "rect_rp",
  "dim"         : [rect_width, rect_smaller_height],
  "color"       : NQColors.light_grey,
  "strokeColor" : NQColors.white,
  "strokeWidth" : 3,
}
export let rect_rp = new Rectangle(params_rect_rp, canvas)

/************************************************
 Init source text field
************************************************/

let text_source_str = `Idea from article by <a href="https://medium.freecodecamp.org/why-you-need-python-environments-and-how-to-manage-them-with-conda-85f155f4353c">Gergely Szerovay</a>.<br>Icon made by Freepik from www.flaticon.com.`
let params_text_source = {
  "pos"            : [1400, 50],
  "textAreaWidth"  : 400,
  "textAreaHeight" : 200,
  "id"             : "text_source",
  "text"           : text_source_str,
  "fontSize"       : 20,
  "textColor"      : NQTextColors.white,
}
export let text_source = new TextObject(params_text_source, canvas)

/************************************************
 Init text field for explanations
************************************************/

var params_tf_exp_1 = {
  pos: [textf_pos_x, textf_pos_y],
  id: "tf_exp_1",
  textAreaWidth: textf_width,
  textAreaHeight: textf_height,
  text: tf_exp_1_str,
  fontSize: 36,
  fontFamily: NQFonts.paragraph,
  rectStroke: "none",
  textColor: NQTextColors.white,
}
export let tf_exp_1 = new TextObject(params_tf_exp_1, canvas)

/************************************************
 Init text fields for environment names
************************************************/

var params_tf_names_1 = {
  pos: [section1_pos_x-0.5*div_step_x, bottom_pos-0.8*div_step_y],
  id: "tf_names_1",
  textAreaWidth: 300,
  textAreaHeight: 100,
  text: "base environment",
  fontSize: 30,
  fontFamily: NQFonts.paragraph,
  rectStroke: "none",
  textColor: NQTextColors.white,
}
export let tf_names_1 = new TextObject(params_tf_names_1, canvas)

var params_tf_names_2 = {
  pos: [section2_pos_x-0.8*div_step_x, bottom_pos-0.8*div_step_y],
  id: "tf_names_2",
  textAreaWidth: 300,
  textAreaHeight: 100,
  text: "additional environment",
  fontSize: 30,
  fontFamily: NQFonts.paragraph,
  rectStroke: "none",
  textColor: NQTextColors.white,
}
export let tf_names_2 = new TextObject(params_tf_names_2, canvas)

var params_tf_names_3 = {
  pos: [section3_pos_x-0.8*div_step_x, bottom_pos-0.8*div_step_y],
  id: "tf_names_3",
  textAreaWidth: 300,
  textAreaHeight: 100,
  text: "R + Python environment",
  fontSize: 30,
  fontFamily: NQFonts.paragraph,
  rectStroke: "none",
  textColor: NQTextColors.white,
}
export let tf_names_3 = new TextObject(params_tf_names_3, canvas)
