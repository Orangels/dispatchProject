function accMul(arg1, arg2) {
  let m = 0;
  const s1 = arg1.toString();
  const s2 = arg2.toString();
  m += s1.split(".").length > 1 ? s1.split(".")[1].length : 0;
  m += s2.split(".").length > 1 ? s2.split(".")[1].length : 0;
  return Number(s1.replace(".", "")) * Number(s2.replace(".", "")) / 10 ** m;
}

export function digitUppercase(n) {
  const fraction = ['角', '分'];
  const digit = ['零', '壹', '贰', '叁', '肆', '伍', '陆', '柒', '捌', '玖'];
  const unit = [['元', '万', '亿'], ['', '拾', '佰', '仟', '万']];
  let num = Math.abs(n);
  let s = '';
  fraction.forEach((item, index) => {
    s += (digit[Math.floor(accMul(num, 10 * 10 ** index)) % 10] + item).replace(/零./, '');
  });
  s = s || '整';
  num = Math.floor(num);
  for (let i = 0; i < unit[0].length && num > 0; i += 1) {
    let p = '';
    for (let j = 0; j < unit[1].length && num > 0; j += 1) {
      p = digit[num % 10] + unit[1][j] + p;
      num = Math.floor(num / 10);
    }
    s = p.replace(/(零.)*零$/, '').replace(/^$/, '零') + unit[0][i] + s;
  }

  return s
    .replace(/(零.)*零元/, '元')
    .replace(/(零.)+/g, '零')
    .replace(/^整$/, '零元整');
}


/**
 * 生成指定区间的随机整数
 * @param min
 * @param max
 * @returns {number}
 */
export function randomNum(min, max) {
  return Math.floor(Math.random() * (max - min) + min);
}

/**
 * 计算提示框的宽度
 * @param str
 * @returns {number}
 */
export function calculateWidth(arr){
  return 30 + arr[0].length*15
}

/**
 * 图片预加载
 * @param arr
 * @constructor
 */
export function preloadingImages(arr) {
  arr.forEach(item=>{
    const img = new Image()
    img.src = item
  })
}


// 原生 utils

// export function randomNum(minNum,maxNum){
//   switch(arguments.length){
//     case 1:
//       return parseInt(Math.random()*minNum+1,10);
//       break;
//     case 2:
//       return parseInt(Math.random()*(maxNum-minNum+1)+minNum,10);
//       break;
//     default:
//       return 0;
//       break;
//   }
// }

export function randomNum_f(minNum,maxNum){
  switch(arguments.length){
    case 1:
      return parseInt(Math.random()*minNum+1,10);
      break;
    case 2:
      return parseInt(Math.random()*(maxNum-minNum+1)+minNum,10);
      break;
    default:
      return 0;
      break;
  }
}


export let _fetch = async (url,data,resolve) =>{
  try {
    let response = await fetch(url,{
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      mode:"cors",
      body: JSON.stringify({
        data
      })
    });
    let json  = await response.json();
    resolve(json);
  } catch(error) {
    console.log(error);
  }
};


export let _download_file = async (url,data,resolve) =>{
  try {
    let response = await fetch(url,{
      method: 'POST',
      headers: {
        'Accept': 'application/json',
        'Content-Type': 'application/json',
      },
      mode:"cors",
      body: JSON.stringify({
        data
      })
    });
    let blob  = await response.blob();
    resolve(blob);
  } catch(error) {
    console.log(error);
  }
};



export function get_state_count(count_arr, state) {
  // [非常, 比较, 正常, 非常不]
  if (state >70  && state <= 100){
    count_arr[0] = count_arr[0] + 1
  }
  if (state > 50 && state <= 70){
    count_arr[1] = count_arr[1] + 1
  }
  if (state > 30 && state <= 50){
    count_arr[2] = count_arr[2] + 1
  }
  if (state >= 0 && state <= 30){
    count_arr[3] = count_arr[3] + 1
  }
  return count_arr
}

export function get_2_float(x) {
  return  Math.round(x*10000)/100;
}

export function show_2_ste(s) {
  return s.toString().substr(0,2)

}

export function get_pie_data(count_arr, name) {

  // [非常, 比较, 正常, 非常不]

  // let aaa = [
  //     {
  //         item: "非常活跃",
  //         count: 30
  //     },
  //     {
  //         item: "非常不活跃",
  //         count: 20
  //     },
  //     {
  //         item: "正常活跃",
  //         count: 27
  //     },
  //     {
  //         item: "比较活跃",
  //         count: 23
  //     },
  // ];

  let total_count = count_arr[0] + count_arr[1] + count_arr[2] + count_arr[3]

  let count_arr_tmp = [
    get_2_float(count_arr[0]/total_count),
    get_2_float(count_arr[1]/total_count),
    get_2_float(count_arr[2]/total_count),
    get_2_float(count_arr[3]/total_count),
  ]

  let return_data = [
    {
      item:`非常${name}`,
      count:count_arr_tmp[0],
    },
    {
      item:`比较${name}`,
      count:count_arr_tmp[1],
    },
    {
      item:`正常${name}`,
      count:count_arr_tmp[2],
    },
    {
      item:`非常不${name}`,
      count:count_arr_tmp[3],
    }
  ]
  // console.log(return_data)
  return return_data

}

export function show_2_int(s){
  return s < 10 ? '0' + s: s;
}

export function deepCopy(obj) {
  // 只拷贝对象
  if (typeof obj !== 'object') return;
  // 根据obj的类型判断是新建一个数组还是一个对象
  let newObj = obj instanceof Array ? [] : {};
  for (let key in obj) {
    // 遍历obj,并且判断是obj的属性才拷贝
    if (obj.hasOwnProperty(key)) {
      // 判断属性值的类型，如果是对象递归调用深拷贝
      newObj[key] = typeof obj[key] === 'object' ? deepCopy(obj[key]) : obj[key];
    }
  }
  return newObj
}


export function diff(obj1,obj2){
  var o1 = obj1 instanceof Object;
  var o2 = obj2 instanceof Object;
  // 判断是不是对象
  if (!o1 || !o2) {
    return obj1 === obj2;
  }

  //Object.keys() 返回一个由对象的自身可枚举属性(key值)组成的数组,
  //例如：数组返回下表：let arr = ["a", "b", "c"];console.log(Object.keys(arr))->0,1,2;
  if (Object.keys(obj1).length !== Object.keys(obj2).length) {
    return false;
  }

  for (var o in obj1) {
    var t1 = obj1[o] instanceof Object;
    var t2 = obj2[o] instanceof Object;
    if (t1 && t2) {
      return diff(obj1[o], obj2[o]);
    } else if (obj1[o] !== obj2[o]) {
      return false;
    }
  }
  return true;
}