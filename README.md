# MPS_vs_BNNS_vs_OUR

## MPS: Metal Performance Shaders
###GoogleNet Inception v3
Time: 128.7[ms](input: 227x227, iPhone 7 Plus)

    func runNetwork(){
    let startTime = CACurrentMediaTime()
    // to deliver optimal performance we leave some resources used in MPSCNN to be released at next call of autoreleasepool,
    // so the user can decide the appropriate time to release this
    autoreleasepool{
    // encoding command buffer
    let commandBuffer = commandQueue.makeCommandBuffer()
    
    // encode all layers of network on present commandBuffer, pass in the input image MTLTexture
    
    Net!.forward(commandBuffer: commandBuffer, sourceTexture: sourceTexture)

    // commit the commandBuffer and wait for completion on CPU
    commandBuffer.commit()
    commandBuffer.waitUntilCompleted()
    
    // display top-5 predictions for what the object should be labelled
    let label = Net!.getLabel()
    predictLabel.text = label
    predictLabel.isHidden = false
    }
    let endTime = CACurrentMediaTime()
    print("time: \(endTime - startTime)");

##BNNS: Basic neural network subroutines
###hogehoge

## OUR 
### Network In NetWork-4layers+BN
Time: 55.7[ms](input: 227x227, iPhone 7 Plus)
### Network In NetWork-5layers+BN
Time: 88.7[ms](input: 227x227, iPhone 7 Plus)
### AlexNet
Time: 108.3[ms](input: 227x227, iPhone 7 Plus)

