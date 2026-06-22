// SPDX-License-Identifier: MIT

pragma solidity ^0.8.20;

contract DeepfakeStorage {

    struct VideoRecord {

        string videoHash;
        string prediction;
        uint256 timestamp;
        address uploader;
    }

    VideoRecord[] public records;

    function storeRecord(
        string memory _videoHash,
        string memory _prediction
    ) public {

        records.push(
            VideoRecord(
                _videoHash,
                _prediction,
                block.timestamp,
                msg.sender
            )
        );
    }

    function getRecord(
        uint256 index
    )
        public
        view
        returns (
            string memory,
            string memory,
            uint256,
            address
        )
    {
        VideoRecord memory record = records[index];

        return (
            record.videoHash,
            record.prediction,
            record.timestamp,
            record.uploader
        );
    }

    function getTotalRecords()
        public
        view
        returns (uint256)
    {
        return records.length;
    }
}