class Queue_len {
    constructor(length){
        this.maxLength = length
        this.data = []
        this.index = -1
    }

    push(items, isArray=true){
        if (this.index > 80000){
            this.index = 8
        }
        if (isArray){
            for (let i=0; i < items.length; i++ ){
                this.data.push(items[i])
                this.index += 1
                if (this.index >= this.maxLength){
                    this.data = this.data.slice(-8)
                }
            }
        }
        else {
            this.data.push(items)
            this.index += 1
            if (this.index >= this.maxLength){
                this.data = this.data.slice(-8)
            }
        }
    }
}

export default Queue_len