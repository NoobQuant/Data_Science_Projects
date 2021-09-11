import {Image} from '../../animlib/src/classes/Image.js'

let width = 1920
let height = 1080

export const svg = d3.select("body").append("div").append("svg")
    // Scalable svg
    .attr('id','bgsvg')    
    .attr('viewBox','0 0 ' + width + ' ' + height)
    .attr('preserveAspectRatio','xMidYMid meet')
    .style("width", '98%')
    .style("height", '98%')
    .style("display", "block")
    .style("position", "absolute");


let params1 = {
    pos: [200,600],
    path: "../pics/html.svg",
    id: "fig_html",
    relSize: [30,30],
    drawType: "fadein"   
}
export let img1 = new Image(params1)

let params2 = {
    pos: [650,600],
    path: "../pics/js.svg",
    id: "fig_js",
    relSize: [30,30],
    drawType: "fadein"   
}
export let img2 = new Image(params2)

let params3 = {
    pos: [1150,600],
    path: "../pics/css.svg",
    id: "fig_css",
    relSize: [30,30],
    drawType: "fadein"   
}
export let img3 = new Image(params3)

let params4 = 
  {
    path: "../pics/browsers.png"
    ,id: "fig_browsers"
    ,pos: [630, 340]
    ,relsize: [40,40]
  }
export var img4 = new Image(params4)

let paramswp_0 = {
  pos: [0,0],
  path: "../pics/webpage_0.svg",
  id: "fig_wp_0",
  relSize: [100,100],
  drawType: "fadein"
}
export var img_wp_0 = new Image(paramswp_0)

let paramswp_1 = {
  pos: [0,0],
  path: "../pics/webpage_1.svg",
  id: "fig_wp_1",
  relSize: [100,100],
  drawType: "fadein"
}
export var img_wp_1 = new Image(paramswp_1)

let paramswp_2 = {
  pos: [0,0],
  path: "../pics/webpage_2.svg",
  id: "fig_wp_2",
  relSize: [100,100],
  drawType: "fadein"
}
export var img_wp_2 = new Image(paramswp_2)

let paramswp_3 = {
  pos: [0,0],
  path: "../pics/webpage_3.svg",
  id: "fig_wp_3",
  relSize: [100,100],
  drawType: "fadein"
}
export var img_wp_3 = new Image(paramswp_3)