$(function(){
   
    $("#b1").click(function(){
        console.log("something");
        $.ajax({
        method: "GET",
        url: "G:\\py\\data-visualization\\sentiment-analysis\\test.py",
        data: {"place" : value},
        dataType: "text",
        success: function(result){
            var data=JSON.parse(result);
            console.log(result);
        }
        
        });
        
        });
});